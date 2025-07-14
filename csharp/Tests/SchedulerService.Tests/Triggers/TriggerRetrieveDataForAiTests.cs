using Dapr.Client;
using Microsoft.Extensions.Logging;
using Moq;
using SchedulerService.Service;
using SchedulerService.Triggers;

namespace SchedulerService.Tests.Triggers;

public class TriggerRetrieveDataForAiTests
{
    private readonly Mock<ILogger<TriggerRetrieveDataForAi>> _mockedLogger = new();
    private readonly Mock<IInvokeDaprService> _mockedInvokeDaprService = new();
    
    [Fact]
    public async Task TriggerRetrieveDataForAi_WhenRun_DaprServiceGetsTriggered()
    {
        // Given
        var trigger = CreateSut();
        // When
        await trigger.RunAsync(CancellationToken.None);
        // Then
        _mockedInvokeDaprService.Verify(x => x.TriggerExportData(It.IsAny<string>(), It.IsAny<CancellationToken>()), Times.Once);
    }

    private TriggerRetrieveDataForAi CreateSut()
    {
        
        var c = new TriggerRetrieveDataForAi(_mockedLogger.Object, _mockedInvokeDaprService.Object);
        return c;
    }
}
