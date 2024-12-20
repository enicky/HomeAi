using Common.ApplicationInsights.Filter;
using Microsoft.ApplicationInsights.Channel;
using Microsoft.ApplicationInsights.DataContracts;
using Microsoft.ApplicationInsights.Extensibility;
using Moq;

namespace DataExporter.Tests.Telemetry;

public class SqlDependencyFilterTests
{
    [Fact]
    public void WhenTypeIsSQL_ProcessingStops(){
        var mockNextProcessor = new Mock<ITelemetryProcessor>();
        var processor = new SqlDependencyFilter(mockNextProcessor.Object);
        var requestTelemetry = new DependencyTelemetry
        {
            Name = "Test Request",

        };
        requestTelemetry.Type = "SQL";       

        // Act
        processor.Process(requestTelemetry);

        // Assert
        // The telemetry should not be passed to the next processor (i.e., it is skipped)
        mockNextProcessor.Verify(p => p.Process(It.IsAny<ITelemetry>()), Times.Never);
    
    }

    [Fact]
    public void WhenOperationDoesNotContainSQL_AndProcessIsSuccess_ProcessingStops(){
        var mockNextProcessor = new Mock<ITelemetryProcessor>();
        var processor = new SqlDependencyFilter(mockNextProcessor.Object);
        var requestTelemetry = new DependencyTelemetry
        {
            Name = "Test Request",

        };
        requestTelemetry.Type = "SSQL";     
        requestTelemetry.Success = true;  

        // Act
        processor.Process(requestTelemetry);

        // Assert
        // The telemetry should not be passed to the next processor (i.e., it is skipped)
        mockNextProcessor.Verify(p => p.Process(It.IsAny<ITelemetry>()), Times.Never);
    
    }

     [Fact]
    public void WhenOperationDoesNotContainSQL_AndProcessIsFalse_ProcessingContinues(){
        var mockNextProcessor = new Mock<ITelemetryProcessor>();
        var processor = new SqlDependencyFilter(mockNextProcessor.Object);
        var requestTelemetry = new DependencyTelemetry
        {
            Name = "Test Request",

        };
        requestTelemetry.Type = "SSQL";     
        requestTelemetry.Success = false;  

        // Act
        processor.Process(requestTelemetry);

        // Assert
        // The telemetry should not be passed to the next processor (i.e., it is skipped)
        mockNextProcessor.Verify(p => p.Process(It.IsAny<ITelemetry>()), Times.Once);
    
    }
}
