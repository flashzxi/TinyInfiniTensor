#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        size_t offset = 0;
        if (!free_block_sizes.empty() && *free_block_sizes.rbegin() >= size) {
            // 从free内分配
            for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
                if (it->second >= size) {
                    offset = it->first;
                    size_t new_offset = it->first + size;
                    size_t remain_size = it->second - size;
                    if (remain_size > 0) {
                        free_block_sizes.erase(free_block_sizes.find(it->second));
                        free_block_sizes.insert(remain_size);
                        it->first = new_offset;
                        it->second = remain_size;
                    } else {
                       free_blocks.erase(it);
                    }
                    break;
                }
            }
        } else {
            offset = peak;
            peak += size;
        }
        used += size;

        return offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================

        size_t free_head = addr;
        size_t free_size = size;
        auto it_prev = std::lower_bound(
                free_blocks.begin(), free_blocks.end(), addr,
                [](const pair<size_t, size_t>& p, size_t value) {
                    return p.first < value;
                }
        );
        if (it_prev != free_blocks.begin()) {
            -- it_prev;
            if (it_prev->first + it_prev->second == addr) {
                free_head = it_prev->first;
                free_size += it_prev->second;
                free_block_sizes.erase(free_block_sizes.find(it_prev->second));
                free_blocks.erase(it_prev);
            }
        }

        auto it_next = std::upper_bound(
                free_blocks.begin(), free_blocks.end(), addr,
                [](size_t value, const pair<size_t, size_t>& p) {
                    return value <= p.first;
                }
        );
        if (it_next != free_blocks.end() && addr + size == it_next->first) {
            free_size += it_next->second;
            free_block_sizes.erase(free_block_sizes.find(it_next->second));
            free_blocks.erase(it_next);
        }
        auto insert_pos = upper_bound(
                free_blocks.begin(), free_blocks.end(), addr,
                [](size_t value, const pair<size_t, size_t>& p) {
                    return value > p.first;
                }
        );
        if (insert_pos == free_blocks.end()) {
            peak = free_head;
        } else {
            free_blocks.insert(insert_pos, {free_head, free_size});
            free_block_sizes.insert(free_size);
        }
        used -= size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
