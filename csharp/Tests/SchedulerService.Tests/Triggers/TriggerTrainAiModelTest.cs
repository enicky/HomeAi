
using Common.Helpers;
using Dapr.Client;
using Microsoft.Extensions.Logging;
using Moq;
using SchedulerService.Triggers;

namespace SchedulerService.Tests.Triggers;

public class TriggerTrainAiModelTest
{
    private readonly Mock<ILogger<TriggerTrainAiModel>> _mockedLogger = new();
    private readonly Mock<DaprClient> _mockedDaprClient = new();
    
    [Fact]
    public async Task TriggerTrainAiModel_WhenRunning_SendMessageOverDapr()
    {
        // Given
        var trigger = CreateSut();
        var cts = new CancellationTokenSource();
        // When
        await trigger.RunAsync(cts.Token);
        // Then
        _mockedDaprClient.Verify(x => x.PublishEventAsync(
            It.Is<string>(q => q == NameConsts.AI_PUBSUB_NAME),
            It.Is<string>(q => q == NameConsts.AI_START_TRAIN_MODEL),
            It.IsAny<CancellationToken>()), Times.Once);

    }

    private TriggerTrainAiModel CreateSut()
    {
        var c = new TriggerTrainAiModel(_mockedLogger.Object, _mockedDaprClient.Object);
        return c;
    }
}
