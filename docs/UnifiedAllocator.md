# Unified Allocator

## 概述

Unified Allocator 是一个高性能、线程安全的跨平台内存分配器框架。该框架通过内存池化技术显著提升了内存管理效率，并提供了适用于不同计算平台的统一接口。

## 设计理念

Unified Allocator 的设计基于以下核心理念：

1. **跨平台统一接口**：通过通用接口支持多种计算平台，包括CPU和各种加速器平台（如CUDA GPU）。
2. **内存池化**：通过重用已分配的内存块，减少频繁分配和释放操作的开销。
3. **智能内存管理**：针对大内存块实现特殊优化策略，优化内存访问性能。
4. **线程安全**：支持多线程环境下的并发内存操作。
5. **内存对齐**：确保所有分配的内存按 128 字节对齐，优化缓存访问。

## 架构设计

Unified Allocator 采用分层架构设计：

1. **BaseUnifiedAllocator**：核心基类，实现通用的内存池管理逻辑和线程安全机制
2. **平台特定实现**：扩展BaseUnifiedAllocator，提供针对不同平台的具体实现
   - **CPUAllocator**：标准CPU内存分配实现
   - 可扩展实现其他平台的分配器（如CUDAUnifiedAllocator等）

这种设计允许在保持统一API的同时，针对不同平台提供优化的内存管理策略。

## 关键特性

### 内存池管理

分配器维护两个内存列表：
- **budgets_**：可用内存池，存储当前未使用的内存块
- **payouts_**：已分配内存池，跟踪当前正在使用的内存块

当请求分配内存时，分配器首先尝试从内存池中查找合适大小的内存块，而不是立即进行新的分配。这显著减少了频繁分配和释放操作的开销，特别是对于重复使用相似大小内存块的应用场景。

### 智能大内存块处理

分配器对大内存块（默认定义为 ≥ 1MB）采用特殊处理：
- 超大内存块（> 2MB）在释放时直接返回系统，而不放入内存池
- 平台特定实现可以提供额外的优化（如在CUDA实现中进行自动预取）

### 内存块元数据

每个分配的内存块都包含元数据，用于跟踪：
- 原始分配指针
- 内存块大小
- 是否为大内存块
- 平台特定信息（如设备ID）
- 最近使用状态
- 所属平台类型

这些元数据支持智能内存管理决策和内存池优化。

### 内存对齐

所有分配的内存按 128 字节对齐，这与主流处理器的缓存行大小兼容，有助于优化内存访问性能，减少缓存未命中。

### 线程安全

通过互斥锁机制确保在多线程环境中的安全操作，支持并发内存分配和释放。

### 多线程性能

支持多线程并发操作，在多线程测试中展现出良好的扩展性。

## 平台支持

### CPU平台（CPUAllocator）

标准CPU内存分配器实现了Unified Allocator接口，使用系统标准的malloc/free函数进行底层内存操作，同时提供内存池优化。

### 可扩展平台

Unified Allocator框架设计支持多种计算平台的扩展实现：

1. **CUDA GPU平台**：可以实现CUDAUnifiedAllocator，利用CUDA统一内存特性
2. **其他加速器平台**：可以针对不同硬件加速器实现专用分配器

## 性能基准测试

各平台实现的性能基准测试可参考相应的性能分析文档：
- [CUDA Unified Allocator 性能基准测试分析](./allocator_benchmark_performance.md)（如已实现）

## 使用场景

Unified Allocator 特别适合以下场景：

1. **跨平台应用开发**：需要在不同硬件平台上提供统一内存管理接口的应用
2. **频繁内存分配/释放**：受益于内存池机制的应用
3. **小到中等大小内存操作**：分配器在这些场景下性能最优
4. **多线程环境**：需要线程安全内存管理的应用
5. **嵌入式系统**：资源受限环境下需要高效内存管理

## API 参考

### 基础分配器（BaseUnifiedAllocator）

基类提供通用接口，但通常不直接实例化，而是使用平台特定实现。

#### 构造函数

```cpp
BaseUnifiedAllocator(Platform platform,
                  const unsigned int size_compare_ratio = 192,
                  const size_t size_drop_threshold = 16);
```

- **platform**：指定平台类型（如HOST、DEVICE等）
- **size_compare_ratio**：内存块大小比较比率（0~256），用于决定内存池中块的重用策略
- **size_drop_threshold**：内存池大小阈值，超过此值时会触发内存块释放

#### 主要方法

##### 内存分配

```cpp
void* malloc(size_t size);
```

分配指定大小的内存块，自动从内存池中查找或创建新的内存块。

##### 内存释放

```cpp
void free(void* ptr);
```

释放指定的内存块，小内存块会放回内存池以供重用，超大内存块会直接释放。

##### 清空内存池

```cpp
void clear();
```

清空内存池，释放所有未使用的内存块。

### CPU分配器（CPUAllocator）

CPU平台的具体实现。

#### 构造函数

```cpp
CPUAllocator(const unsigned int size_compare_ratio = 192,
             const size_t size_drop_threshold = 16);
```

参数与BaseUnifiedAllocator相同。

## 使用示例

### CPU分配器基本使用

```cpp
#include "utils/unified_allocator.h"

// 创建CPU分配器实例
CPUAllocator allocator;

// 分配内存
void* ptr = allocator.malloc(1024);  // 分配 1KB 内存

// 使用内存
// ...

// 释放内存
allocator.free(ptr);
```

### 多线程环境

```cpp
// 分配器是线程安全的，可以在多线程环境中使用
CPUAllocator shared_allocator;

// 线程函数
void thread_function() {
    void* ptr = shared_allocator.malloc(4096);
    // 使用内存
    // ...
    shared_allocator.free(ptr);
}

// 创建多个线程
std::thread t1(thread_function);
std::thread t2(thread_function);
// ...

// 等待线程完成
t1.join();
t2.join();
// ...
```

## 平台扩展指南

要为新平台实现Unified Allocator，需要：

1. 从BaseUnifiedAllocator派生新的平台特定类
2. 实现platformMalloc()和platformFree()方法
3. 根据平台特性提供额外优化（可选）

示例实现框架：

```cpp
class NewPlatformAllocator : public BaseUnifiedAllocator
{
public:
    NewPlatformAllocator(const unsigned int size_compare_ratio = 192,
                         const size_t size_drop_threshold = 16)
        : BaseUnifiedAllocator(Platform::NEW_PLATFORM, size_compare_ratio, size_drop_threshold)
    {
        // 平台特定初始化
    }

    ~NewPlatformAllocator() override
    {
        // 平台特定清理
    }

protected:
    void* platformMalloc(size_t size) override
    {
        // 平台特定的内存分配实现
    }

    void platformFree(void* ptr) override
    {
        // 平台特定的内存释放实现
    }
    
    // 可选：添加平台特定的优化方法
};
```

## 性能优化建议

1. **内存大小选择**：
   - 对于频繁分配释放的小内存块，充分利用内存池机制
   - 对于大内存块，尽量减少分配释放频率

2. **平台选择**：
   - 根据应用场景选择最合适的平台分配器
   - 在混合计算环境中，可以使用多种分配器实例管理不同类型的内存

3. **内存池参数调优**：
   - 对于内存使用模式已知的应用，可以调整 `size_compare_ratio` 和 `size_drop_threshold` 参数
   - 内存密集型应用可能需要更大的 `size_drop_threshold` 值

4. **对齐考虑**：
   - 分配器自动处理 128 字节对齐，无需手动对齐
   - 但在访问模式上，尽量遵循缓存友好的访问模式

## 限制与注意事项

1. **平台限制**：
   - 不同平台实现可能有特定的限制和要求
   - 参考各平台实现的具体文档

2. **内存开销**：
   - 每个分配的内存块都有额外的元数据开销
   - 内存池机制可能会暂时保留一些未使用的内存

3. **析构行为**：
   - 分配器析构时会检查是否有未释放的内存，如有则输出警告
   - 建议在应用程序结束前显式释放所有分配的内存

4. **错误处理**：
   - 分配失败时返回 nullptr
   - 释放非法指针时会输出错误日志并尝试安全释放

5. **多线程环境**：
   - 虽然分配器本身是线程安全的，但使用分配的内存时仍需考虑线程安全问题

## 内部实现细节

### 内存块选择算法

当从内存池中查找合适的内存块时，分配器使用以下策略：

1. 遍历内存池中的所有块
2. 如果找到大小大于等于请求大小，且满足 `(block_size * size_compare_ratio) >> 8 <= requested_size` 的块，则使用该块
3. 如果内存池已满（超过 `size_drop_threshold`），则释放最大或最小的块，取决于哪个与请求大小差距更大

### 内存对齐实现

分配器在分配内存时会分配额外的空间用于元数据和对齐：

```cpp
size_t allocation_size = size + sizeof(MemoryBlock) + GRYFLUX_MEMORY_ALIGN;
```

然后使用 `alignPtr` 函数计算对齐的用户指针位置：

```cpp
void* user_ptr = alignPtr((unsigned char*)original_ptr + sizeof(MemoryBlock), GRYFLUX_MEMORY_ALIGN);
```

### 元数据存储

元数据存储在用户指针前面的内存位置，可以通过简单的指针算术找到：

```cpp
MemoryBlock* metadata_location = (MemoryBlock*)((unsigned char*)user_ptr - sizeof(MemoryBlock));
```

同时，元数据也在全局注册表中维护，用于快速查找和管理。

## 总结

Unified Allocator 是一个高性能、跨平台的内存分配框架，通过内存池化和统一接口设计，为不同计算平台提供了高效的内存管理解决方案。其模块化架构支持轻松扩展到新平台，同时保持一致的API和优化的性能特性。基准测试结果表明，在小到中等大小内存操作和频繁分配释放场景中，该分配器相比标准内存分配具有明显的性能优势。
