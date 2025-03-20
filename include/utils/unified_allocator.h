/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************************************************************/
#pragma once

#include <list>
#include <string>
#include <utility>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <cstdlib>
#include <memory>
#include "utils/logger.h"

// 定义内存对齐大小 (128字节，兼容AGX Orin L2缓存线大小)
#define GRYFLUX_MEMORY_ALIGN 128

// 大内存块阈值 (1 MB)
#define LARGE_MEMORY_THRESHOLD (1 * 1024 * 1024)

// 平台类型
enum class Platform
{
    HOST,
    DEVICE
};

// Aligns a pointer to the specified number of bytes
template <typename _Tp>
static inline _Tp *alignPtr(_Tp *ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp *)(((size_t)ptr + n - 1) & -n);
}

// 跟踪内存块的元数据
struct MemoryBlock
{
    void *original_ptr;              // 原始分配的指针
    size_t size;                     // 分配的大小
    bool is_large;                   // 是否为大内存块
    std::atomic<int> device_id;      // 内存当前所在设备ID
    std::atomic<bool> recently_used; // 最近是否被使用
    Platform platform;               // 内存所属平台

    // 防止拷贝构造和赋值操作
    MemoryBlock() : device_id(0), recently_used(false), platform(Platform::HOST) {}
    MemoryBlock(const MemoryBlock &) = delete;
    MemoryBlock &operator=(const MemoryBlock &) = delete;
};

// 内存块注册表，用于管理和跟踪所有分配的内存
class MemoryRegistry
{
private:
    std::mutex registry_mutex_;
    std::unordered_map<void *, MemoryBlock *> memory_blocks_;

public:
    MemoryRegistry() {}

    ~MemoryRegistry()
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        memory_blocks_.clear();
    }

    void registerBlock(void *user_ptr, MemoryBlock *block)
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        memory_blocks_[user_ptr] = block;
    }

    void unregisterBlock(void *user_ptr)
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = memory_blocks_.find(user_ptr);
        if (it != memory_blocks_.end())
        {
            memory_blocks_.erase(it);
        }
    }

    MemoryBlock *getBlock(void *user_ptr)
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = memory_blocks_.find(user_ptr);
        if (it != memory_blocks_.end())
        {
            return it->second;
        }
        return nullptr;
    }
};

// 基础内存分配器接口
class BaseUnifiedAllocator
{
protected:
    unsigned int size_compare_ratio_; // 0~256
    size_t size_drop_threshold_;
    std::mutex allocator_mutex_;
    std::list<std::pair<size_t, void *>> budgets_;
    std::list<std::pair<size_t, void *>> payouts_;
    MemoryRegistry registry_;
    Platform platform_;

public:
    BaseUnifiedAllocator(Platform platform,
                  const unsigned int size_compare_ratio = 192,
                  const size_t size_drop_threshold = 16)
        : size_compare_ratio_(size_compare_ratio),
          size_drop_threshold_(size_drop_threshold),
          platform_(platform)
    {
    }

    virtual ~BaseUnifiedAllocator()
    {
        clear();
        std::lock_guard<std::mutex> lock(allocator_mutex_);
        if (!this->payouts_.empty())
        {
            LOG.error("[ALLOCATOR] FATAL ERROR! Allocator destroyed while memory still in use");
            for (auto &item : payouts_)
            {
                void *ptr = item.second;
                LOG.error("[ALLOCATOR] %p still in use", ptr);
            }
        }
    }

    // 分配内存
    void *malloc(size_t size)
    {
        // 将大小向上取整到内存对齐边界
        size = (size + GRYFLUX_MEMORY_ALIGN - 1) & ~(GRYFLUX_MEMORY_ALIGN - 1);

        void *ptr = nullptr;
        bool is_new_allocation = false;

        {
            std::lock_guard<std::mutex> lock(allocator_mutex_);

            // 尝试从内存池中查找合适大小的内存块
            auto it = budgets_.begin();
            auto it_max = budgets_.begin();
            auto it_min = budgets_.begin();

            for (; it != budgets_.end(); ++it)
            {
                size_t bs = it->first;

                // 大小适合且在可接受比率范围内
                if (bs >= size && ((bs * size_compare_ratio_) >> 8) <= size)
                {
                    ptr = it->second;
                    budgets_.erase(it);
                    payouts_.push_back(std::make_pair(bs, ptr));

                    // 更新内存块元数据
                    MemoryBlock *block = registry_.getBlock(ptr);
                    if (block)
                    {
                        block->recently_used = true;
                        LOG.trace("[ALLOCATOR] Reuse memory %p, size is %zu", ptr, bs);
                    }
                    return ptr;
                }

                if (it != budgets_.end())
                {
                    if (bs < it_min->first)
                        it_min = it;
                    if (bs > it_max->first)
                        it_max = it;
                }
            }

            // 如果内存池已满，释放一些内存块
            if (!budgets_.empty() && budgets_.size() >= size_drop_threshold_)
            {
                if (it_max->first < size)
                {
                    // 释放最小的内存块
                    ptr = it_min->second;
                    MemoryBlock *block = registry_.getBlock(ptr);
                    if (block)
                    {
                        registry_.unregisterBlock(ptr);
                        platformFree(block->original_ptr);
                        delete block;
                    }
                    budgets_.erase(it_min);
                }
                else if (it_min->first > size)
                {
                    // 释放最大的内存块
                    ptr = it_max->second;
                    MemoryBlock *block = registry_.getBlock(ptr);
                    if (block)
                    {
                        registry_.unregisterBlock(ptr);
                        platformFree(block->original_ptr);
                        delete block;
                    }
                    budgets_.erase(it_max);
                }
            }

            // 需要分配新内存
            is_new_allocation = true;
        }

        // 分配过程不需要锁住整个分配器
        if (is_new_allocation)
        {
            // 分配新内存
            ptr = allocateMemory(size);

            if (ptr)
            {
                std::lock_guard<std::mutex> lock(allocator_mutex_);
                payouts_.push_back(std::make_pair(size, ptr));
            }
        }

        return ptr;
    }

    // 释放内存
    void free(void *ptr)
    {
        if (!ptr)
            return;

        bool found = false;
        size_t size = 0;

        {
            std::lock_guard<std::mutex> lock(allocator_mutex_);

            // 查找内存块
            auto it = payouts_.begin();
            for (; it != payouts_.end(); ++it)
            {
                if (it->second == ptr)
                {
                    size = it->first;
                    payouts_.erase(it);
                    found = true;
                    break;
                }
            }

            if (found)
            {
                // 检查是否为大内存块
                MemoryBlock *block = registry_.getBlock(ptr);
                if (block)
                {
                    // 非常大的内存块直接释放，不放入内存池
                    if (size > LARGE_MEMORY_THRESHOLD * 2)
                    {
                        registry_.unregisterBlock(ptr);
                        platformFree(block->original_ptr);
                        delete block;
                    }
                    else
                    {
                        // 其他内存块放回内存池
                        block->recently_used = false;
                        budgets_.push_back(std::make_pair(size, ptr));
                        LOG.trace("[ALLOCATOR] Recycle memory %p, size is %zu", ptr, size);
                    }
                }
                return;
            }
        }

        if (!found)
        {
            LOG.error("[ALLOCATOR] FATAL ERROR! Allocator get wild pointer %p", ptr);
            // 尝试直接释放
            MemoryBlock *block = registry_.getBlock(ptr);
            if (block)
            {
                registry_.unregisterBlock(ptr);
                platformFree(block->original_ptr);
                delete block;
            }
        }
    }

    // 清空内存池
    void clear()
    {
        std::lock_guard<std::mutex> lock(allocator_mutex_);
        for (auto &item : budgets_)
        {
            void *ptr = item.second;
            MemoryBlock *block = registry_.getBlock(ptr);
            if (block)
            {
                registry_.unregisterBlock(ptr);
                platformFree(block->original_ptr);
                delete block;
            }
        }
        budgets_.clear();
    }

    // 获取平台类型
    Platform getPlatform() const
    {
        return platform_;
    }

protected:
    // 平台特定的内存分配实现（由子类实现）
    void *allocateMemory(size_t size)
    {
        void *original_ptr = nullptr;

        // 分配额外空间用于元数据和对齐
        size_t allocation_size = size + sizeof(MemoryBlock) + GRYFLUX_MEMORY_ALIGN;

        // 分配原始内存
        original_ptr = platformMalloc(allocation_size);
        if (!original_ptr)
        {
            LOG.error("[ALLOCATOR] CPU memory allocation failed");
            return nullptr;
        }

        // 计算对齐的用户指针位置
        void *user_ptr = alignPtr((unsigned char *)original_ptr + sizeof(MemoryBlock), GRYFLUX_MEMORY_ALIGN);

        // 创建并初始化元数据
        MemoryBlock *block = new MemoryBlock;
        block->original_ptr = original_ptr;
        block->size = allocation_size;
        block->is_large = (size >= LARGE_MEMORY_THRESHOLD);
        block->device_id.store(0); // CPU总是设备0
        block->recently_used.store(true);
        block->platform = platform_;

        // 在内存中初始化元数据
        MemoryBlock *metadata_location = (MemoryBlock *)((unsigned char *)user_ptr - sizeof(MemoryBlock));
        metadata_location->original_ptr = original_ptr;
        metadata_location->size = allocation_size;
        metadata_location->is_large = (size >= LARGE_MEMORY_THRESHOLD);
        metadata_location->device_id.store(0);
        metadata_location->recently_used.store(true);
        metadata_location->platform = platform_;

        registry_.registerBlock(user_ptr, block);
        return user_ptr;
    }

    // 平台特定的内存释放实现（由子类实现）
    virtual void platformFree(void *ptr) = 0;

    // 平台特定的内存分配实现（由子类实现）
    virtual void *platformMalloc(size_t size) = 0;
};

// CPU内存分配器实现
class CPUAllocator : public BaseUnifiedAllocator
{
public:
    CPUAllocator(const unsigned int size_compare_ratio = 192,
                 const size_t size_drop_threshold = 16)
        : BaseUnifiedAllocator(Platform::HOST, size_compare_ratio, size_drop_threshold)
    {
    }

    ~CPUAllocator() override = default;

protected:
    void *platformMalloc(size_t size) override
    {
        return std::malloc(size);
    }

    void platformFree(void *ptr) override
    {
        if (ptr)
        {
            std::free(ptr);
        }
    }
};
