using System.Diagnostics;
using System.IO.Abstractions;
using System.Threading;
using System.Threading.Tasks;
using Common.Factory;
using Common.Models.AI;
using Common.Services;
using DataExporter.Controllers;
using DataExporter.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit;

namespace DataExporter.Tests.ControllerTests;

public class InfluxController_TraceParentTests : IClassFixture<TestSetup>
{
    private readonly ServiceProvider _serviceProvider;
    private readonly IConfiguration _configuration;
    private readonly Mock<IInfluxDbService> _mockedInfluxDbService = new();
    private readonly Mock<IFileService> _mockedFileService = new();
    private readonly Mock<IDaprClientWrapper> _mockedDaprClient = new();
    private readonly Mock<ILocalFileService> _localFileService = new();
    private readonly Mock<ILogger<InfluxController>> _mockedLogger = new();
    private readonly Mock<ICleanupService> _mockedCleanupService = new();
    private readonly Mock<IFileSystem> _mockedFileSystem = new();

    public InfluxController_TraceParentTests(TestSetup testSetup)
    {
        _serviceProvider = testSetup.ServiceProvider;
        _configuration = _serviceProvider.GetRequiredService<IConfiguration>();
    }

    [Fact]
    public async Task RetrieveData_WithTraceParent_SetsParentId()
    {
        // Arrange
        var traceParent = "00-4bf92f3577b34da6a3ce929d0e0e4733-00f067aa0ba902b7-00";
        var evt = new StartDownloadDataEvent { TraceParent = traceParent };
        var cts = new CancellationTokenSource();
        var expected = new List<Common.Models.Influx.InfluxRecord> { new() };
        _mockedInfluxDbService.Setup(x => x.QueryAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<CancellationToken>())).ReturnsAsync(expected);
        _mockedFileService.Setup(x => x.RetrieveParsedFile(It.IsAny<string>(), It.IsAny<string>())).ReturnsAsync("dummy-file.csv");
        _localFileService.Setup(x => x.ReadFromFile(It.IsAny<string>())).Returns(expected);
        _mockedCleanupService.Setup(x => x.Cleanup(It.IsAny<List<Common.Models.Influx.InfluxRecord>>(), It.IsAny<List<Common.Models.Influx.InfluxRecord>>())).Returns(expected);
        _mockedFileSystem.Setup(x => x.File.Exists(It.IsAny<string>())).Returns(true);

        var controller = new InfluxController(
            _mockedInfluxDbService.Object,
            _mockedFileService.Object,
            _configuration,
            _mockedCleanupService.Object,
            _mockedDaprClient.Object,
            _localFileService.Object,
            _mockedLogger.Object
        )
        {
            FileSystem = _mockedFileSystem.Object
        };

        var tcs = new TaskCompletionSource<Activity>();
        using var listener = new ActivityListener
        {
            ShouldListenTo = _ => true,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStarted = activity =>
            {
                if (activity.OperationName == "InfluxController.RetrieveData")
                    tcs.TrySetResult(activity);
            },
            ActivityStopped = _ => { }
        };
        ActivitySource.AddActivityListener(listener);

        // Act
        await controller.RetrieveData(evt, cts.Token);
#pragma warning disable xUnit1031 // Do not use blocking task operations in test method
        var capturedActivity = await Task.WhenAny(tcs.Task, Task.Delay(1000)) == tcs.Task ? tcs.Task.Result : null;
#pragma warning restore xUnit1031 // Do not use blocking task operations in test method

        // Assert
        Assert.NotNull(capturedActivity);
        Assert.Equal(traceParent, capturedActivity.ParentId);
    }
}
