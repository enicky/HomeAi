using Common.ApplicationInsights.Filter;
using Microsoft.ApplicationInsights.Channel;
using Microsoft.ApplicationInsights.DataContracts;
using Microsoft.ApplicationInsights.Extensibility;
using Moq;

namespace SchedulerService.Tests.Telemetry;

public class HangfireRequestFilterTests
{
    [Fact]
    public void WhenOperationNameContainsHangfire_ProcessingStops()
    {
        var mockNextProcessor = new Mock<ITelemetryProcessor>();
        var processor = new HangfireRequestFilter(mockNextProcessor.Object);
        var requestTelemetry = new RequestTelemetry
        {
            Name = "Test Request",

        };
        requestTelemetry.Context.Operation.Name = "hangfire";

        // Act
        processor.Process(requestTelemetry);

        // Assert
        // The telemetry should not be passed to the next processor (i.e., it is skipped)
        mockNextProcessor.Verify(p => p.Process(It.IsAny<ITelemetry>()), Times.Never);

    }

    [Fact]
    public void WhenOperationNameDoesNotContainsHangfire_ProcessingContinues()
    {
        var mockNextProcessor = new Mock<ITelemetryProcessor>();
        var processor = new HangfireRequestFilter(mockNextProcessor.Object);
        var requestTelemetry = new RequestTelemetry
        {
            Name = "Test Request",

        };
        requestTelemetry.Context.Operation.Name = "SomethingElse";

        // Act
        processor.Process(requestTelemetry);

        // Assert
        // The telemetry should not be passed to the next processor (i.e., it is skipped)
        mockNextProcessor.Verify(p => p.Process(It.IsAny<ITelemetry>()), Times.Once);

    }
}
