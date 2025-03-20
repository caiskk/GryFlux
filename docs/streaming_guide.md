# 📊 GryFlux 流式处理框架指南

<div align="center">
  <img src="https://img.shields.io/badge/版本-1.0-blue.svg" alt="版本">
  <img src="https://img.shields.io/badge/语言-C++-orange.svg" alt="语言">
  <img src="https://img.shields.io/badge/许可证-MIT-green.svg" alt="许可证">
</div>

<div align="center">
  <p><i>高性能数据流处理系统 - 为连续数据流应用提供强大支持</i></p>
</div>

---

## 📑 目录

- [1. 流式处理框架概述](#1-流式处理框架概述)
- [2. 架构组件](#2-架构组件)
- [3. 使用指南](#3-使用指南)
- [4. 计算图构建详解](#4-计算图构建详解)
- [5. 性能分析与优化](#5-性能分析与优化)
- [6. 示例应用](#6-示例应用)
- [7. 故障排除](#7-故障排除)
- [8. 总结](#8-总结)

---

## 1. 框架概述

GryFlux 流式处理框架是一个基于任务图的高性能数据处理系统，特别适用于需要连续处理数据流的场景，如视频分析、传感器数据处理等。它提供了一个灵活的架构，允许开发者构建高效的数据处理流水线。

### 1.1 核心特点

<div align="center">
<table>
  <tr>
    <th align="center" width="250">特点</th>
    <th align="center">说明</th>
  </tr>
  <tr>
    <td align="center"><b>🧩 基于任务的模块化设计</b></td>
    <td>将复杂的处理逻辑分解为可重用的任务模块</td>
  </tr>
  <tr>
    <td align="center"><b>⚡ 并行处理能力</b></td>
    <td>利用线程池实现任务并行执行</td>
  </tr>
  <tr>
    <td align="center"><b>🔄 生产者-消费者模式</b></td>
    <td>支持异步数据流处理</td>
  </tr>
  <tr>
    <td align="center"><b>📊 性能分析工具</b></td>
    <td>内置执行时间统计和性能监控</td>
  </tr>
  <tr>
    <td align="center"><b>🔗 任务依赖管理</b></td>
    <td>自动处理任务之间的依赖关系</td>
  </tr>
</table>
</div>

---

## 2. 架构组件

### 2.1 核心组件

<div align="center">
<table>
  <tr>
    <th align="center" width="250">组件</th>
    <th align="center">功能描述</th>
  </tr>
  <tr>
    <td align="center"><b>🌊 StreamingPipeline</b></td>
    <td>流式处理管道，管理数据流和任务执行</td>
  </tr>
  <tr>
    <td align="center"><b>🏗️ PipelineBuilder</b></td>
    <td>构建和执行任务图</td>
  </tr>
  <tr>
    <td align="center"><b>⏱️ TaskScheduler</b></td>
    <td>调度和执行任务</td>
  </tr>
  <tr>
    <td align="center"><b>📍 TaskNode</b></td>
    <td>表示任务图中的一个节点</td>
  </tr>
  <tr>
    <td align="center"><b>🧵 ThreadPool</b></td>
    <td>管理工作线程</td>
  </tr>
  <tr>
    <td align="center"><b>🏭 DataProducer/DataConsumer</b></td>
    <td>数据生产者和消费者接口</td>
  </tr>
  <tr>
    <td align="center"><b>📋 TaskRegistry</b></td>
    <td>管理和注册处理任务</td>
  </tr>
</table>
</div>

---

## 3. 使用指南

### 3.1 创建流式处理应用

以下是创建基本流式处理应用的步骤：

#### 1. 初始化任务注册表

```cpp
// 创建任务注册表实例
GryFlux::TaskRegistry taskRegistry;
```

#### 2. 注册处理任务

```cpp
// 注册各种处理任务
taskRegistry.registerTask<GryFlux::ObjectDetector>("objectDetection");
taskRegistry.registerTask<GryFlux::XfeatExtractor>("objectTracking");
// 更多任务...
```

#### 3. 创建流式处理管道

```cpp
// 创建处理管道，指定线程数
GryFlux::StreamingPipeline pipeline(10); // 使用10个线程
```

#### 4. 设置输出节点 ID

```cpp
// 设置输出节点标识符
pipeline.setOutputNodeId("output");
```

#### 5. 定义计算图构建函数

```cpp
void buildStreamingComputeGraph(std::shared_ptr<GryFlux::PipelineBuilder> builder,
                                std::shared_ptr<GryFlux::DataObject> input,
                                const std::string &outputId,
                                GryFlux::TaskRegistry &taskRegistry)
{
    // 输入节点
    auto inputNode = builder->addInput("input", input);

    // 构建任务节点
    auto task1Node = builder->addTask("task1",
                                      taskRegistry.getProcessFunction("task1"),
                                      {inputNode});

    // 更多任务节点...

    // 输出节点
    builder->addTask(outputId,
                     taskRegistry.getProcessFunction("output"),
                     {task1Node, /* 其他输入... */});
}
```

#### 6. 设置计算图

```cpp
// 设置处理器函数
pipeline.setProcessor([&taskRegistry](std::shared_ptr<GryFlux::PipelineBuilder> builder,
                                      std::shared_ptr<GryFlux::DataObject> input,
                                      const std::string &outputId)
{
    buildStreamingComputeGraph(builder, input, outputId, taskRegistry);
});
```

#### 7. 启动管道处理

```cpp
// 启动处理管道
pipeline.start();
```

#### 8. 创建数据生产者和消费者

```cpp
// 创建控制变量和数据处理组件
std::atomic<bool> running(true);
GryFlux::TestImageProducer producer(pipeline, running);
GryFlux::TestConsumer consumer(pipeline, running);

// 启动生产者和消费者
producer.start();
consumer.start();
```

#### 9. 等待处理完成并清理资源

```cpp
// 等待处理完成
producer.join();
consumer.join();
pipeline.stop();
```

### 3.2 实现自定义处理任务

创建自定义处理任务需要继承 ProcessingTask 类：

```cpp
class MyCustomTask : public GryFlux::ProcessingTask
{
public:
    MyCustomTask() {}

    virtual std::shared_ptr<GryFlux::DataObject> process(
        const std::vector<std::shared_ptr<GryFlux::DataObject>> &inputs) override
    {
        // 处理输入数据
        // ...

        // 返回处理结果
        return std::make_shared<GryFlux::DataObject>();
    }
};

// 注册自定义任务
taskRegistry.registerTask<MyCustomTask>("myCustomTask");
```

### 3.3 实现自定义数据生产者

实现自定义数据生产者需要继承`DataProducer`类：

```cpp
class MyDataProducer : public GryFlux::DataProducer
{
public:
    MyDataProducer(GryFlux::StreamingPipeline& pipeline, std::atomic<bool>& running, Allocator *allocator)
        : DataProducer(pipeline, running, allocator) {}

protected:
    void run() override {
        while (running.load()) {
            // 创建或获取数据
            auto data = createData();

            // 添加到管道
            if (!addData(data)) {
                break;
            }

            // 控制生产速率
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::shared_ptr<GryFlux::DataObject> createData() {
        // 创建数据对象
        return std::make_shared<GryFlux::DataObject>();
    }
};
```

### 3.4 实现自定义数据消费者

实现自定义数据消费者需要继承`DataConsumer`类：

```cpp
class MyDataConsumer : public GryFlux::DataConsumer
{
public:
    MyDataConsumer(GryFlux::StreamingPipeline& pipeline, std::atomic<bool>& running, Allocator *allocator)
        : DataConsumer(pipeline, running, allocator), processedFrames_(0) {}

    int getProcessedFrames() const { return processedFrames_; }

protected:
    void run() override {
        while (shouldContinue()) {
            std::shared_ptr<GryFlux::DataObject> data;

            // 尝试获取处理结果
            if (getData(data)) {
                // 处理输出数据
                processData(data);
                processedFrames_++;
            } else {
                // 避免CPU空转
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    }

    void processData(std::shared_ptr<GryFlux::DataObject> data) {
        // 处理输出数据
    }

private:
    int processedFrames_;
};
```

---

## 4. 计算图构建详解

### 4.1 计算图概念

<div align="center">
  <p>
    <img src="./resource/dag.svg" alt="计算图示例" width="500">
  </p>
  <p><i>计算图是一个有向无环图(DAG)，其中节点代表计算任务，边表示数据依赖关系</i></p>
</div>

在流式处理框架中，`PipelineBuilder`类负责构建和管理这个计算图，而 `TaskScheduler` 则负责根据依赖关系执行任务。

### 4.2 PipelineBuilder API 详解

PipelineBuilder 类提供了两个关键方法用于构建计算图：

#### 4.2.1 addInput 方法

```cpp
std::shared_ptr<TaskNode> addInput(const std::string &id, std::shared_ptr<DataObject> data);
```

<div align="center">
<table>
  <tr>
    <th align="center" width="150">参数</th>
    <th align="center">说明</th>
  </tr>
  <tr>
    <td align="center"><code>id</code></td>
    <td>输入节点的唯一标识符</td>
  </tr>
  <tr>
    <td align="center"><code>data</code></td>
    <td>输入数据对象</td>
  </tr>
  <tr>
    <td align="center"><b>返回值</b></td>
    <td>创建的输入节点，可被后续任务引用</td>
  </tr>
  <tr>
    <td align="center"><b>作用</b></td>
    <td>创建一个输入节点，作为计算图的数据源</td>
  </tr>
</table>
</div>

#### 4.2.2 addTask 方法

```cpp
std::shared_ptr<TaskNode> addTask(
    const std::string &id,
    std::function<std::shared_ptr<DataObject>(const std::vector<std::shared_ptr<DataObject>> &)> func,
    const std::vector<std::shared_ptr<TaskNode>> &inputs);
```

<div align="center">
<table>
  <tr>
    <th align="center" width="150">参数</th>
    <th align="center">说明</th>
  </tr>
  <tr>
    <td align="center"><code>id</code></td>
    <td>
      <p><b>任务节点的唯一标识符，用于在计算图中引用该任务</b></p>
      <ul>
        <li>每个任务必须有唯一的 ID</li>
        <li>输出节点的 ID 应与 setOutputNodeId 设置的 ID 相匹配</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td align="center"><code>func</code></td>
    <td>
      <p><b>处理函数，接受一组输入数据对象，返回一个输出数据对象</b></p>
      <ul>
        <li>函数签名：std::shared_ptr&lt;DataObject&gt;(const std::vector&lt;std::shared_ptr&lt;DataObject&gt;&gt; &)</li>
        <li>通常通过 TaskRegistry.getProcessFunction()获取注册的处理函数</li>
        <li>也可以使用 lambda 表达式定义内联处理逻辑</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td align="center"><code>inputs</code></td>
    <td>
      <p><b>该任务的输入节点列表</b></p>
      <ul>
        <li>可以是输入节点(addInput 的返回值)或其他任务节点(addTask 的返回值)</li>
        <li>列表顺序决定了输入数据在处理函数中的索引顺序</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td align="center"><b>返回值</b></td>
    <td>创建的任务节点，可被后续任务引用作为输入</td>
  </tr>
  <tr>
    <td align="center"><b>作用</b></td>
    <td>创建一个处理任务节点，并定义其与其他节点的依赖关系</td>
  </tr>
</table>
</div>

### 4.3 构建复杂计算图示例

以下是一个构建复杂计算图的详细示例：

```cpp
void buildAdvancedComputeGraph(std::shared_ptr<GryFlux::PipelineBuilder> builder,
                             std::shared_ptr<GryFlux::DataObject> input,
                             const std::string &outputId,
                             GryFlux::TaskRegistry &taskRegistry)
{
    // 创建输入节点
    auto inputNode = builder->addInput("input", input);

    // 分支1：预处理 -> 特征提取
    auto preprocessNode = builder->addTask("preprocess",
                                         taskRegistry.getProcessFunction("imagePreprocess"),
                                         {inputNode});

    auto featureNode = builder->addTask("featureExtract",
                                       taskRegistry.getProcessFunction("featureExtraction"),
                                       {preprocessNode});

    // 分支2：对象检测
    auto detectionNode = builder->addTask("objectDetection",
                                         taskRegistry.getProcessFunction("yoloDetector"),
                                         {inputNode});

    // 结合两个分支的结果进行对象跟踪
    auto trackingNode = builder->addTask("objectTracking",
                                        taskRegistry.getProcessFunction("tracker"),
                                        {featureNode, detectionNode});

    // 内联处理函数示例（使用lambda）
    auto statisticsNode = builder->addTask("statistics",
        [](const std::vector<std::shared_ptr<GryFlux::DataObject>> &inputs) -> std::shared_ptr<GryFlux::DataObject> {
            auto trackingResult = inputs[0];
            // 计算统计数据
            auto stats = std::make_shared<GryFlux::DataObject>();
            // ... 处理逻辑
            return stats;
        },
        {trackingNode});

    // 多输入节点示例
    auto visualizationNode = builder->addTask("visualization",
                                             taskRegistry.getProcessFunction("visualize"),
                                             {inputNode, detectionNode, trackingNode});

    // 最终输出节点（合并多个结果）
    builder->addTask(outputId,
                    taskRegistry.getProcessFunction("resultAggregator"),
                    {trackingNode, statisticsNode, visualizationNode});
}
```

<div align="center">
  <p>
    <img src="./resource/example_graph.svg" alt="复杂计算图示例" width="600">
  </p>
  <p><i>复杂计算图示例可视化</i></p>
</div>

### 4.4 addTask 高级用法

#### 4.4.1 处理多输入任务

```cpp
// 三输入示例：融合RGB图像、深度图和热成像数据
auto fusionNode = builder->addTask("sensorFusion",
                                  taskRegistry.getProcessFunction("multiModalFusion"),
                                  {rgbNode, depthNode, thermalNode});
```

#### 4.4.2 重用中间结果

```cpp
// 同一节点被多个后续任务使用
auto featureNode = builder->addTask("features", featureExtractor, {inputNode});

auto classifierNode = builder->addTask("classifier", classifier, {featureNode});
auto segmentationNode = builder->addTask("segmentation", segmentor, {featureNode});
auto keypointNode = builder->addTask("keypoints", keypointDetector, {featureNode});
```

#### 4.4.3 实现条件分支（静态）

```cpp
// 在构建时决定使用哪个模型
std::shared_ptr<TaskNode> detectorNode;
if (useHighAccuracyModel) {
    detectorNode = builder->addTask("detectorHQ", hqDetector, {inputNode});
} else {
    detectorNode = builder->addTask("detectorFast", fastDetector, {inputNode});
}
```

### 4.5 计算图构建最佳实践

<div class="best-practices">
  <div align="center">
    <table>
      <tr>
        <th align="center" width="50">🔢</th>
        <th align="center" width="200">最佳实践</th>
        <th align="center">说明</th>
      </tr>
      <tr>
        <td align="center">1</td>
        <td align="center"><b>保持任务粒度适中</b></td>
        <td>
          <ul>
            <li>过细的粒度会增加调度开销</li>
            <li>过粗的粒度会减少并行性</li>
          </ul>
        </td>
      </tr>
      <tr>
        <td align="center">2</td>
        <td align="center"><b>管理依赖关系</b></td>
        <td>
          <ul>
            <li>避免创建循环依赖（框架不支持）</li>
            <li>尽量减少任务之间的不必要依赖</li>
          </ul>
        </td>
      </tr>
      <tr>
        <td align="center">3</td>
        <td align="center"><b>任务命名约定</b></td>
        <td>
          <ul>
            <li>使用清晰、有意义的 ID</li>
            <li>使用命名模式表示相关任务（如 image_preprocess、image_crop）</li>
          </ul>
        </td>
      </tr>
      <tr>
        <td align="center">4</td>
        <td align="center"><b>重用计算结果</b></td>
        <td>
          <ul>
            <li>将共享的预处理或特征提取步骤作为单独的任务</li>
            <li>多个后续任务可共享同一个前置任务的结果</li>
          </ul>
        </td>
      </tr>
      <tr>
        <td align="center">5</td>
        <td align="center"><b>错误处理</b></td>
        <td>
          <ul>
            <li>在处理函数中妥善处理异常</li>
            <li>对于可能失败的操作，返回适当的错误状态而非抛出异常</li>
          </ul>
        </td>
      </tr>
      <tr>
        <td align="center">6</td>
        <td align="center"><b>性能优化</b></td>
        <td>
          <ul>
            <li>将计算密集型任务拆分为多个并行任务</li>
            <li>使用性能分析工具识别瓶颈任务</li>
          </ul>
        </td>
      </tr>
    </table>
  </div>
</div>

---

## 5. 性能分析与优化

### 5.1 启用性能分析

框架内置性能分析功能，可以跟踪任务执行时间和处理吞吐量:

```cpp
// 启用性能分析
pipeline.enableProfiling(true);
```

启用性能分析后，框架将收集并输出以下信息:

- 总处理项目数
- 每个任务的执行时间
- 平均处理时间
- 处理速率

### 5.2 性能优化建议

<div align="center">
<table>
  <tr style="background-color: #4CAF50; color: white; text-align: center;">
    <th align="center" width="150">优化方向</th>
    <th align="center" width="300">具体建议</th>
    <th align="center">预期效果</th>
  </tr>
  <tr>
    <td align="center"><b>🧵 线程管理</b></td>
    <td>根据系统CPU核心数和工作负载调整线程数</td>
    <td>平衡资源利用和上下文切换开销</td>
  </tr>
  <tr>
    <td align="center"><b>🧩 任务粒度</b></td>
    <td>避免过细或过粗的任务划分，寻找最佳平衡点</td>
    <td>提高并行效率，减少调度开销</td>
  </tr>
  <tr>
    <td align="center"><b>📦 数据传输</b></td>
    <td>尽可能使用数据引用和智能指针，减少拷贝</td>
    <td>降低内存使用和CPU开销</td>
  </tr>
  <tr>
    <td align="center"><b>📊 队列设置</b></td>
    <td>根据内存限制和生产消费速率调整队列大小</td>
    <td>防止内存溢出，平衡处理延迟</td>
  </tr>
  <tr>
    <td align="center"><b>🔗 依赖优化</b></td>
    <td>减少不必要的任务依赖，避免串行瓶颈</td>
    <td>提高并行度，减少等待时间</td>
  </tr>
</table>
</div>

---

## 6. 示例应用

**example_stream.cpp** 展示了一个无人机视频处理系统的实现，它包括以下功能：

1. 视频帧对象检测
2. 特征提取
3. 图像压缩
4. 前景/背景分离
5. 熵编码

通过流式处理框架，这些功能可以高效并行执行，实现实时视频分析和处理。

<div align="center">
  <p>
    <img src="./resource/example_stream.svg" alt="无人机视频处理系统" width="700">
  </p>
  <p><i>无人机视频处理系统计算图</i></p>
</div>

---

## 7. 故障排除

### 7.1 常见问题及解决方案

<div align="center">
<table>
  <tr style="background-color: #f8d7da; color: #721c24; text-align: center;">
    <th align="center" width="150">问题类型</th>
    <th align="center" width="300">可能原因</th>
    <th align="center">解决方法</th>
  </tr>
  <tr>
    <td align="center"><b>⚠️ 任务执行异常</b></td>
    <td>任务依赖不正确，输入数据无效</td>
    <td>检查任务依赖关系，验证输入数据有效性</td>
  </tr>
  <tr>
    <td align="center"><b>⏱️ 性能瓶颈</b></td>
    <td>某些任务执行时间过长</td>
    <td>使用性能分析工具识别，优化或进一步并行化任务</td>
  </tr>
  <tr>
    <td align="center"><b>💾 内存占用过高</b></td>
    <td>输入队列过大，资源未及时释放</td>
    <td>控制队列大小，及时释放不再使用的资源</td>
  </tr>
  <tr>
    <td align="center"><b>⏳ 数据处理延迟</b></td>
    <td>处理速度跟不上数据产生速度</td>
    <td>增加处理线程，优化算法，降低数据生产率</td>
  </tr>
  <tr>
    <td align="center"><b>🔒 任务死锁</b></td>
    <td>资源竞争或依赖循环</td>
    <td>检查依赖关系，确保没有循环依赖</td>
  </tr>
</table>
</div>

### 7.2 调试技巧

<div align="center">
<table>
  <tr>
    <th align="center" width="200">调试方法</th>
    <th align="center">说明</th>
  </tr>
  <tr>
    <td align="center"><b>🔍 启用详细日志</b></td>
    <td>设置日志级别为 DEBUG 或 TRACE，获取更详细的执行信息</td>
  </tr>
  <tr>
    <td align="center"><b>🧪 单元测试</b></td>
    <td>为每个任务编写单元测试，验证其独立功能正确性</td>
  </tr>
  <tr>
    <td align="center"><b>📊 性能分析</b></td>
    <td>使用内置性能分析工具识别瓶颈任务和资源消耗</td>
  </tr>
  <tr>
    <td align="center"><b>🔄 简化计算图</b></td>
    <td>从简单计算图开始，逐步添加复杂性，定位问题</td>
  </tr>
  <tr>
    <td align="center"><b>📝 状态检查点</b></td>
    <td>在关键节点添加状态检查，验证中间数据正确性</td>
  </tr>
</table>
</div>

---

## 8. 总结

GryFlux流式处理框架提供了构建高性能数据处理应用的强大工具。通过任务图和并行执行，它能够有效处理高吞吐量的数据流，同时保持代码的模块化和可维护性。

### 8.1 主要优势

<div align="center">
<table>
  <tr>
    <th align="center" width="200">优势</th>
    <th align="center">说明</th>
  </tr>
  <tr>
    <td align="center"><b>🧩 模块化设计</b></td>
    <td>促进代码复用，提高可维护性</td>
  </tr>
  <tr>
    <td align="center"><b>⚡ 并行执行</b></td>
    <td>提高处理效率，充分利用多核处理器</td>
  </tr>
  <tr>
    <td align="center"><b>📊 灵活的任务图</b></td>
    <td>支持复杂数据流和处理逻辑</td>
  </tr>
  <tr>
    <td align="center"><b>📈 内置性能分析</b></td>
    <td>助力识别和解决性能瓶颈</td>
  </tr>
  <tr>
    <td align="center"><b>🔄 异步处理</b></td>
    <td>高效处理连续数据流</td>
  </tr>
</table>
</div>

### 8.2 适用场景

- **视频分析系统**：实时处理视频流，执行对象检测、跟踪和识别
- **传感器数据处理**：处理来自多个传感器的数据流，执行融合和分析
- **实时监控系统**：连续处理和分析监控数据，生成警报和报告
- **数据流转换管道**：将数据从一种格式转换为另一种格式，同时执行清洗和验证

适当使用框架提供的性能分析工具，可以帮助识别和解决性能瓶颈，进一步优化应用性能。通过遵循本指南中的最佳实践，开发者可以充分利用流式处理框架的强大功能，构建高效、可靠的数据处理应用。

<div align="center">
  <p style="margin-top: 50px; color: #666;">
    <i>© 2025 GryFlux Gricc</i>
  </p>
</div>
