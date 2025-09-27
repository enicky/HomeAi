using Common.Helpers;
using Common.Models.AI;
using Common.Models.Responses;
using Dapr.Client;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using SchedulerService.Controllers;
using SchedulerService.Triggers;

namespace SchedulerService.Tests.Controllers;

public class DaprControllerTests
{
    private readonly Mock<ILogger<DaprController>> _mockedLogger = new();
    private readonly Mock<DaprClient> _mockedDaprClient = new();
    private readonly Mock<IServiceProvider> _mockedServiceProvider = new();

    [Fact]
    public void TestReply_Test()
    {
        var controller = CreateSut();

        var p = new RetrieveDataResponse()
        {
            GeneratedFileName = "test",
            StartAiProcess = true,
            Success = true
        };

        controller.TestReply(p);
        _mockedLogger.Verify(x => x.Log(
            LogLevel.Information,
            It.IsAny<EventId>(),
            It.Is<It.IsAnyType>((v, t) => true),
            It.IsAny<Exception>(),
            (Func<It.IsAnyType, Exception?, string>)It.IsAny<object>()), Times.AtMost(2));
    }

    [Fact]
    public void TestReply_WhenPassingNullAsARgument_WeDontLog2Statements()
    {
        // Given
        var controller = CreateSut();
        // When
        controller.TestReply(null);
        // Then
        _mockedLogger.Verify(x => x.Log(
            LogLevel.Information,
            It.IsAny<EventId>(),
            It.Is<It.IsAnyType>((v, t) => true),
            It.IsAny<Exception>(),
            (Func<It.IsAnyType, Exception?, string>)It.IsAny<object>()), Times.Once);
    }

    [Fact]
    public async Task DownloadDataHasFinished_WhenResponseWasNotSuccessfull_WeLogWarning()
    {
        // Given
        var controller = CreateSut();
        var p = new RetrieveDataResponse()
        {
            GeneratedFileName = "test",
            StartAiProcess = true,
            Success = false
        };
        // When
        await controller.DownloadDataHasFinished(p);
        // Then
        _mockedLogger.Verify(x => x.Log(
            LogLevel.Warning,
            It.IsAny<EventId>(),
            It.Is<It.IsAnyType>((v, t) => true),
            It.IsAny<Exception>(),
            (Func<It.IsAnyType, Exception?, string>)It.IsAny<object>()), Times.Once);
    }
    
    [Fact]
    public async Task DownloadDataHasFinished_WhenResponseWasSuccess_AndStartAiProcessIsTrue_SendDaprMessage()
    {
        // Given
        var controller = CreateSut();
        var p = new RetrieveDataResponse()
        {
            GeneratedFileName = "test",
            StartAiProcess = true,
            Success = true
        };
        // When
        await controller.DownloadDataHasFinished(p);
        // Then
        _mockedDaprClient.Verify(x => x.PublishEventAsync(It.Is<string>( x => x == NameConsts.AI_PUBSUB_NAME),
                                    It.Is<string>(x => x == NameConsts.AI_START_DOWNLOAD_DATA),
                                    It.IsAny<string>(),
                                    It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task DownloadDataHasFinished_WhenResponseWasNull_JustReturn()
    {
        // Given
        var controller = CreateSut();
        // When
        await controller.DownloadDataHasFinished(null);
        // Then
        _mockedDaprClient.Verify(x => x.PublishEventAsync(It.Is<string>( x => x == NameConsts.AI_PUBSUB_NAME),
                                    It.Is<string>(x => x == NameConsts.AI_START_DOWNLOAD_DATA),
                                    It.IsAny<CancellationToken>()), Times.Never);
    
    }

    [Fact]
    public async Task DownloadDataHasFinished_WhenResponseWasSuccess_AndStartAiProcessIsFalse_DoNotSendDaprMessage()
    {
        // Given
        var controller = CreateSut();
        var p = new RetrieveDataResponse()
        {
            GeneratedFileName = "test",
            StartAiProcess = false,
            Success = true
        };
        // When
        await controller.DownloadDataHasFinished(p);
        // Then
        _mockedDaprClient.Verify(x => x.PublishEventAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<CancellationToken>()), Times.Never);
    }

    [Fact]
    public async Task AiDownloadFinishedStartTraining_WhenReceiving_ForwardMessageOnDapr()
    {
        // Given
        var controller = CreateSut();
        var cts = new CancellationTokenSource();
        // When
        await controller.AiDownloadFinishedStartTraining(cts.Token);
        // Then
        _mockedDaprClient.Verify(a => a.PublishEventAsync(It.Is<string>(x => x == NameConsts.AI_PUBSUB_NAME),
            It.Is<string>(x => x == NameConsts.AI_START_TRAIN_MODEL), It.IsAny<StartTrainModelEvent>(), It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task AiFinishedTrainingModel_WhenWeReceiveAResponse_AndModelPathIsNull_WeDoNothing()
    {
        // Given
        var controller = CreateSut();

        // When
        await controller.AiFinishedTrainingModel(new TrainAiModelResponse()
        {
            Success = true
        });
        // Then
        _mockedDaprClient.Verify(x => x.PublishEventAsync(It.Is<string>(xx => xx == NameConsts.AI_PUBSUB_NAME),
            It.Is<string>(xx => xx == NameConsts.AI_START_UPLOAD_MODEL), It.IsAny<CancellationToken>()), Times.Never);
    }

    [Fact] 
    public async Task AiFinishedTrainingModel_WhenWeReceiveAResponse_AndModelPathIsValid_WePublishMessage(){
        var controller = CreateSut();
        var modelResponse = new TrainAiModelResponse()
        {
            Success = true,
            ModelPath = "test"
        };
        await controller.AiFinishedTrainingModel(modelResponse);
        _mockedDaprClient.Verify(x => x.PublishEventAsync(It.Is<string>(xx => xx == NameConsts.AI_PUBSUB_NAME),
            It.Is<string>(xx => xx == NameConsts.AI_START_UPLOAD_MODEL), It.IsAny<StartUploadModel>(),
             It.IsAny<CancellationToken>()), Times.Once);

    }


    private DaprController CreateSut()
    {
        var mockLogger = new Mock<ILogger<TriggerRetrieveDataForAi>>();
        var sc = new ServiceCollection();
        sc.AddSingleton(mockLogger.Object);
        var serviceProvider = new ServiceCollection().BuildServiceProvider();
        
        return new DaprController(sc.BuildServiceProvider(), _mockedLogger.Object, _mockedDaprClient.Object);
    }
}
