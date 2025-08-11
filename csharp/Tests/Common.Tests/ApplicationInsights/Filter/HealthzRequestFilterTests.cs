using Common.ApplicationInsights.Filter;
using Microsoft.ApplicationInsights.Channel;
using Microsoft.ApplicationInsights.DataContracts;
using Microsoft.ApplicationInsights.Extensibility;
using Moq;
using Xunit;

namespace Common.Tests.ApplicationInsights.Filter;

public class HealthzRequestFilterTests
{
    [Theory]
    [InlineData("GET /healthz")]
    [InlineData("GET /health")]
    [InlineData("GET /ready")]
    [InlineData("GET /live")]
    [InlineData("GET /api/healthz")] // Contains "healthz"
    public void Process_FiltersHealthRequests_DoesNotCallNext(string operationName)
    {
        // Arrange
        var mockTelemetry = new Mock<ITelemetry>();
        var context = new TelemetryContext { Operation = { Name = operationName } };
        mockTelemetry.Setup(t => t.Context).Returns(context);
        var next = new Mock<ITelemetryProcessor>();
        var filter = new HealthzRequestFilter(next.Object);

        // Act
        filter.Process(mockTelemetry.Object);

        // Assert
        next.Verify(n => n.Process(It.IsAny<ITelemetry>()), Times.Never);
    }

    [Theory]
    [InlineData("GET /api/realendpoint")]
    [InlineData("POST /something")]
    [InlineData("")]
    [InlineData(null)]
    public void Process_NonHealthRequests_CallsNext(string operationName)
    {
        // Arrange
        var mockTelemetry = new Mock<ITelemetry>();
        var context = new TelemetryContext { Operation = { Name = operationName } };
        mockTelemetry.Setup(t => t.Context).Returns(context);
        var next = new Mock<ITelemetryProcessor>();
        var filter = new HealthzRequestFilter(next.Object);

        // Act
        filter.Process(mockTelemetry.Object);

        // Assert
        next.Verify(n => n.Process(mockTelemetry.Object), Times.Once);
    }
}
