from collections import OrderedDict
from typing import Callable, List, Tuple, Union

import torch

from colossalai.inference.struct import Sequence
from colossalai.logging import get_dist_logger
from colossalai.utils import get_current_device

logger = get_dist_logger(__name__)


class BatchBucket:
    """Container for a batch of Sequences, which is used to manage the batch of sequences.

    Attrs:
        sequences_dict (OrderedDict[int, Sequence]): Map sequence uid to sequence struct
            seq_uid -> Sequence
        sequences_indexes (OrderedDict[int, int]): Map sequence uid to index in the batch
            seq_uid -> index in the batch (indexing used in sequence_lengths and block_tables)
        sequence_lengths (torch.Tensor): Length of each sequence in the batch.
            The size of the tensor is (max_batch_size,)
        block_tables (torch.Tensor): Block table of each sequence in the batch
            The size of the tensor is (max_batch_size, max_blocks_per_seq)
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        max_batch_size,
        max_length,
        block_size,
        kv_max_split_num,
        fd_interm_tensor=None,
        device=None,
        dtype=torch.float16,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_length = max_length  # in + out len
        self.block_size = block_size
        self.kv_max_split_num = kv_max_split_num  # Hint used for flash decoding
        self.fd_interm_tensor = fd_interm_tensor
        self.device = device or get_current_device()
        self.dtype = dtype

        self._current_batch_size = 0
        self._sequences_dict = OrderedDict()
        self._sequences_indexes = OrderedDict()  # deque(maxlen=self.max_batch_size)
        self._sequence_lengths = torch.zeros((self.max_batch_size,), dtype=torch.int32)
        self._sequence_lengths_helper = torch.zeros_like(self._sequence_lengths)
        max_blocks_per_seq = (self.max_length + block_size - 1) // block_size
        self._block_tables = torch.full((self.max_batch_size, max_blocks_per_seq), -1, dtype=torch.int32)
        self._block_tables_helper = torch.full_like(self._block_tables, -1)

    @property
    def is_empty(self):
        return self._current_batch_size == 0

    @property
    def current_batch_size(self):
        return self._current_batch_size

    @property
    def available_batch_size(self):
        return self.max_batch_size - self._current_batch_size

    @property
    def block_tables(self):
        return self._block_tables

    @property
    def seq_lengths(self):
        return self._sequence_lengths

    @property
    def seqs_ids(self):
        return list(self._sequences_dict.keys())

    @property
    def seqs_li(self):
        return list(self._sequences_dict.values())

    @property
    def is_compact(self):
        assert len(self._sequences_dict) == len(self._sequences_indexes), "BatchBucket indexing is not consistent"
        return (
            len(self._sequences_dict)
            == torch.nonzero(self._sequence_lengths).view(-1).numel()
            == torch.nonzero(self._block_tables[:, 0] >= 0).numel()
        )

    def add_seq(
        self,
        seq: Sequence,
        alloc_block_table: torch.Tensor = None,
        alloc_block_table_fn: Callable[[torch.Tensor, int], None] = None,
    ) -> Union[torch.Tensor, None]:
        """Add a single sequence to the batch.
        User could opt to provide either a block table or a function to allocate block tables.

        Args:
            seq (Sequence): The sequence to be added to the batch
            alloc_block_table (torch.Tensor): The block tables to be copied and used for the sequence
            alloc_block_table_fn (Callable[[torch.Tensor, int], None]): The function to allocate blocks for the sequence,
                which is expected to reserve blocks and update status of kv-cache manager.

        Returns:
            block_table (torch.Tensor): The block table of the added sequence, used for block allocation in kv-cache manager.
                None if the sequence cannot be added.
        """
        block_table = None
        # TODO might consider sorting by length
        if self._current_batch_size < self.max_batch_size:
            self._sequences_dict[seq.request_id] = seq
            self._sequences_indexes[seq.request_id] = self._current_batch_size
            self._sequence_lengths[self._current_batch_size] = seq.sentence_len
            # NOTE the added seq still require block table allocation by kvcache manager
            block_table = self._block_tables[self._current_batch_size - 1]
            if alloc_block_table is not None:
                # copy block ids from provided block tables
                self._block_tables[self._current_batch_size - 1] = alloc_block_table
            elif alloc_block_table_fn:
                alloc_block_table_fn(block_table, self._sequence_lengths[self._current_batch_size - 1].item())
            self._current_batch_size += 1
        return block_table

    def add_seqs(
        self,
        seqs: List[Sequence],
        alloc_block_tables: torch.Tensor = None,
        alloc_block_tables_fn: Callable[[torch.Tensor, torch.Tensor], None] = None,
    ) -> Union[torch.Tensor, None]:
        """Add a list of sequences to the batch.
        User could opt to provide either block tables or a function to allocate block tables.

        Args:
            seqs (List[Sequence]): The sequences to be added to the batch
            alloc_block_tables (torch.Tensor): The block tables to be copied and used for the sequence
            alloc_block_table_fn (Callable[[torch.Tensor, torch.Tensor], None]): The function to allocate blocks for multiple sequences,
                which is expected to reserve blocks and update status of kv-cache manager.

        Returns:
            block_tables (torch.Tensor): The block tables of the added sequences, used for block allocation in kv-cache manager.
                None if the sequences cannot be added.
        """

        assert (
            alloc_block_tables is None or alloc_block_tables_fn is None
        ), "`alloc_block_tables` and `alloc_block_tables_fn` cannot be provided at the same time"

        num_seqs_to_add = min(self.max_batch_size - self._current_batch_size, len(seqs))
        block_tables = None
        if num_seqs_to_add > 0:
            for i, seq in enumerate(seqs[:num_seqs_to_add]):
                self._sequences_dict[seq.request_id] = seq
                self._sequences_indexes[seq.request_id] = self._current_batch_size + i
            # TODO external (rename): modify Sequence.sentence_len to seq_len
            self._sequence_lengths[
                self._current_batch_size : self._current_batch_size + num_seqs_to_add
            ] = torch.tensor([seq.sentence_len for seq in seqs[:num_seqs_to_add]], dtype=torch.int32)
            # NOTE block tables to be updated by kvcache manager
            block_tables = self._block_tables[self._current_batch_size : self._current_batch_size + num_seqs_to_add]
            if alloc_block_tables is not None:
                # copy block ids from provided block tables
                self._block_tables[
                    self._current_batch_size : self._current_batch_size + num_seqs_to_add
                ] = alloc_block_tables
            elif alloc_block_tables_fn:
                alloc_block_tables_fn(
                    block_tables,
                    self._sequence_lengths[self._current_batch_size : self._current_batch_size + num_seqs_to_add],
                )

            self._current_batch_size += num_seqs_to_add
            seqs[:] = seqs[num_seqs_to_add:]

        return block_tables

    def pop_seq_update_batch(
        self, request_id: int, free_block_table_fn: Callable[[torch.Tensor], None] = None
    ) -> Tuple[Sequence, Union[torch.Tensor, None]]:
        """Pop a single sequence by id from the batch, and update the batch bucket status.

        Args:
            request_id (int): The uid of the sequence
            free_block_table_fn (Callable): The function to free the block table of a sequence,
                if not provided, then we have to release the block table manually after calling this method

        Returns:
            A tuple of: seq (Sequence): The target sequence
            and block_table (torch.Tensor): block table of the target sequence indicating corresponding blocks,
                none if the sequence is not found or free_block_table_fn is provided.
        """
        seq: Sequence = self._sequences_dict.get(request_id)
        block_table = None
        if seq is not None:
            assert request_id in self._sequences_indexes, "Inconsistency in BatchBucket indexing"
            self._sequences_dict.pop(request_id)
            seq_b_idx = self._sequences_indexes.get(request_id)

            if self.current_batch_size > 1:
                # replace seq length of the target seq with that of the last seq in the batch
                last_seq_id = next(
                    (uid for uid, index in self._sequences_indexes.items() if index == self.current_batch_size - 1),
                    None,
                )
                assert last_seq_id is not None
                last_seq_b_idx = self._sequences_indexes[last_seq_id]
                self._sequences_indexes[last_seq_id] = seq_b_idx
                self._sequence_lengths[seq_b_idx] = self._sequence_lengths[last_seq_b_idx]
                self._sequence_lengths[last_seq_b_idx].fill_(0)
                # free the block table of the seq, or return a copy of the block table (to be processed outside)
                if free_block_table_fn:
                    free_block_table_fn(self._block_tables[seq_b_idx])
                else:
                    block_table = self._block_tables[seq_b_idx].detach().clone()
                # replace block table of the target seq with that of the last seq in the batch
                self._block_tables[seq_b_idx] = self._block_tables[last_seq_b_idx]
                self._block_tables[last_seq_b_idx].fill_(-1)
            else:
                if free_block_table_fn:
                    free_block_table_fn(self._block_tables[0])
                else:
                    block_table = self._block_tables[0].detach().clone()
                self._sequence_lengths[0].fill_(0)
                self._block_tables[0].fill_(-1)
            self._sequences_indexes.pop(request_id)
            self._current_batch_size -= 1

        return seq, block_table

    def pop_seqs(
        self, request_ids: List[int], free_block_table_fn: Callable[[torch.Tensor], None] = None
    ) -> Tuple[List[Sequence], List[torch.Tensor]]:
        """Iteratively pop a list of sequences by uid.

        Args:
            request_ids (List[int]): The uids of the sequences
            free_block_table_fn (Callable): The function to free the block table of a sequence,
                if not provided, then we have to release the block table manually after calling this method
        Returns:
            A tuple of: seqs (List[Sequence]): The target sequences
            and block_tables (List[torch.Tensor]): block tables of the target sequences indicating corresponding blocks
        """
        seqs = []
        block_tables = []
        for request_id in request_ids:
            seq, block_table = self.pop_seq_update_batch(request_id, free_block_table_fn)
            if seq is not None:
                seqs.append(seq)
            if block_table is not None:
                block_tables.append(block_table)
        return seqs, block_tables

    def pop_n_seqs(
        self, n: int, free_block_table_fn: Callable[[torch.Tensor], None] = None
    ) -> Tuple[List[Sequence], List[torch.Tensor]]:
        """Pop the first n sequences in the batch (FIFO).
        If n is greater than the current batch szie, pop all the sequences in the batch.

        Returns:
            A tuple of: seqs (List[Sequence]): The target sequences
            and block_tables (List[torch.Tensor]): block tables of the target sequences indicating corresponding blocks
        """
        # NOTE Prevent calling this method multiple times in a single step
        # NOTE might consider: return a list of tensors or a 2D tensor
        seqs = []
        block_tables = []
        n = min(n, self.current_batch_size)
        for _ in range(n):
            _, seq = self._sequences_dict.popitem(last=False)
            seq_b_idx = self._sequences_indexes.pop(seq.request_id)
            if free_block_table_fn:
                free_block_table_fn(self.block_tables[seq_b_idx])
            else:
                block_tables.append(self.block_tables[seq_b_idx].detach().clone())
            seqs.append(seq)
        if not self.is_compact:
            self.make_compact()
        return seqs, block_tables

    def make_compact(self) -> None:
        # Clean and Compress the batch based on its sequences dict.
        # Namely,compress sequences to the front and clean the seq lengths and block tables tensors.
        # NOTE Prevent calling this method multiple times in a single step
        if self.is_compact:
            return
        valid_seq_ids = self._sequences_dict.keys()
        valid_num = len(valid_seq_ids)
        valid_indexes = [self._sequences_indexes[seq_id] for seq_id in valid_seq_ids]
        assert valid_num == len(self._sequences_indexes), "BatchBucket indexing is not consistent"
        self._sequence_lengths_helper[:valid_num] = self._sequence_lengths[valid_indexes]
        self._sequence_lengths[:] = self._sequence_lengths_helper[:]
        self._block_tables_helper[:valid_num, :] = self.block_tables[valid_indexes]
        self.block_tables[:] = self._block_tables_helper[:]
        self._sequence_lengths_helper.fill_(0)
        self._block_tables_helper.fill_(-1)
        self._current_batch_size = valid_num

    def pop_finished(
        self, free_block_table_fn: Callable[[torch.Tensor], None] = None
    ) -> Tuple[List[Sequence], List[torch.Tensor]]:
        finished_seqs = []
        finished_block_tables = []
        for request_id in self._sequences_indexes:
            seq: Sequence = self._sequences_dict.get(request_id)
            if seq.check_finish():
                finished_seqs.append(seq)

        if len(finished_seqs) < self.max_batch_size // 4:
            for seq in finished_seqs:
                _, block_table = self.pop_seq_update_batch(seq.request_id, free_block_table_fn)
                if block_table is not None:
                    finished_block_tables.append(block_table)
        else:
            for seq in finished_seqs:
                seq_id = seq.request_id
                block_table = self.block_tables[self._sequences_indexes[seq_id]]
                self._sequences_dict.pop(seq_id)
                self._sequences_indexes.pop(seq_id)
                if free_block_table_fn:
                    free_block_table_fn(block_table)
                else:
                    finished_block_tables.append(block_table.detach().clone())
            self.make_compact()
        return finished_seqs, finished_block_tables

    # TODO arg type not support beam search sampling yet
    def append_batch_tokens(self, tokens: torch.Tensor) -> None:
        """Append a batch of tokens to the sequences in the batch"""
        assert self.current_batch_size == tokens.size(0), "Batch size mismatch"

        if self.current_batch_size > 0:
            tokens = tokens.tolist()
            for i, seq in enumerate(self._sequences_dict.values()):
                curr_tokens = tokens[i] if isinstance(tokens[i], list) else [tokens[i]]
                seq.output_token_id += curr_tokens
                seq.check_finish()
            self._sequence_lengths[: self.current_batch_size] += 1

    def clear(self, free_block_tables_fn: Callable[[torch.Tensor], None]) -> List[int]:
        """Clear all the sequences in the batch.

        free_block_tables_fn (Callable): The function to free the block tables of all the sequences in the batch,
        """
        seqs = list(self._sequences_dict.values())
        self._sequences_dict.clear()
        self._sequences_indexes.clear()
        free_block_tables_fn(self.block_tables, self._current_batch_size)
        self.block_tables.fill_(-1)
        self._sequence_lengths.fill_(0)
        self._current_batch_size = 0
        return seqs

    def merge(self, other: "BatchBucket") -> List[int]:
        """Merge the sequences in the other batch into the current batch.
        Merge as possible as the current batch can, if it does not have available spaces
        holding all the sequences in the other batch

        Usage:
            New incoming sequence added to prefil batch
                prefill bb curr batch size < prefil_ratio * prefill bb max batch size
            New incoming sequence added to prefil batch
                prefill bb curr batch size < prefil_ratio * prefill bb max batch size
            New incoming sequence added to prefil batch
                prefill bb curr batch size == prefil_ratio * prefill bb max batch size
            Pause Decoding
            Prefill
            Move sequences in prefill bb => decoding bb
            Put back the out-of-volume sequences into the running pool

        Returns:
            unmerged_ids (List[int]): a list of sequence uids that are not merged into the current batch
        """
        unmerged_ids = []
        num_seqs_to_merge = min(self.available_batch_size, other.current_batch_size)
        if num_seqs_to_merge > 0:
            seqs, block_tables_li = other.pop_n_seqs(num_seqs_to_merge)
            block_tables = torch.stack(block_tables_li)
            self.add_seqs(seqs, alloc_block_tables=block_tables)
            unmerged_ids = other.seqs_ids
        return unmerged_ids

    ########## The following methods are expected to be used in modeling ###########

    # For compatibility.
    # NOTE: Treat this as an assumed method for determining the stage of the batch.
    @property
    def is_prompts(self) -> bool:
        assert len(self._sequences_dict) > 0, "No sequence in the batch"
        first_seq = list(self._sequences_dict.values())[0]
        if first_seq.output_len == 0:
            return True
        return False

    def get_1D_inputs(self) -> torch.Tensor:
        assert len(self._sequences_dict) > 0, "No sequence in the batch"
        tokens_li = []
        first_seq = list(self._sequences_dict.values())[0]
        if first_seq.output_len == 0:
            # Assume prefill stage
            assert all(
                seq.output_len == 0 for seq in self._sequences_dict.values()
            ), "Sequence stage (Prefill/Decoding) must be the same in the batch"
            for seq in self._sequences_dict.values():
                tokens_li.extend(seq.input_token_id)
        else:
            # Assume decoding stage
            assert all(
                seq.output_len > 0 for seq in self._sequences_dict.values()
            ), "Sequence stage (Prefill/Decoding) must be the same in the batch"
            for seq in self._sequences_dict.values():
                tokens_li.append(seq.output_token_id[-1])

        return torch.tensor(tokens_li, dtype=torch.long, device=self.device)

    # For compatibility
    def get_block_table_tensor(self) -> torch.Tensor:
        assert self.is_compact  # Debug usage
        block_table = self.block_tables[: self.current_batch_size]
        return block_table.to(device=self.device)

    # For compatibility
    def get_sequence_lengths(self) -> torch.Tensor:
        assert self.is_compact  # Debug usage
        sequence_lengths = self.seq_lengths[: self.current_batch_size]
        return sequence_lengths.to(device=self.device)

    # FOR compatibility
    @property
    def fd_inter_tensor(self) -> None:
        assert self.fd_interm_tensor is not None, "fd_interm_tensor is not provided"
        return self.fd_interm_tensor
