using Common.ApplicationInsights.Initializers;
using Microsoft.ApplicationInsights.DataContracts;

namespace DataExporter.Tests.Telemetry;

public class CustomTelemetryInitializerTests
{
    [Fact]
    public void InitializeTests()
    {
        var customPropertyValue = "TestValue";
        var initializer = new CustomTelemetryInitializer(customPropertyValue);
        var exceptionTelemetry = new ExceptionTelemetry(new Exception("Test exception"));

        // Act
        initializer.Initialize(exceptionTelemetry);

        // Assert
        Assert.NotNull(exceptionTelemetry.Context.Cloud.RoleName);
        Assert.Equal(customPropertyValue, exceptionTelemetry.Context.Cloud.RoleName);
    }
}
