using Common.Factory;
using Common.Helpers;
using Common.Models.AI;
using Common.Models.Influx;
using Common.Models.Responses;
using Common.Services;
using DataExporter.Controllers;
using DataExporter.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using System.Diagnostics;
using Xunit.Abstractions;

namespace DataExporter.Tests.ControllerTests;

public class InfluxControllerTests : IClassFixture<TestSetup>
{
    private readonly ServiceProvider _serviceProvider;
    private readonly IConfiguration _configuration;
    private readonly ITestOutputHelper _output;
    private readonly Mock<IInfluxDbService> _mockedInfluxDbService = new();
    private readonly Mock<IFileService> _mockedFileService = new();
    private readonly Mock<IDaprClientWrapper> _mockedDaprClient = new();
    private readonly Mock<ILocalFileService> _localFileService = new();
    private readonly Mock<ILogger<InfluxController>> _mockedLogger = new();

    public InfluxControllerTests(TestSetup testSetup, ITestOutputHelper output)
    {
        _serviceProvider = testSetup.ServiceProvider;
        _configuration = _serviceProvider.GetRequiredService<IConfiguration>();
        _output = output;
        _output.WriteLine("Starting in constructor");
    }

    [Fact]
    public async Task WhenRetrieveDataHasBeenCalled_AndWeHaveData_WeSendDataThroughDapr()
    {
        var cts = new CancellationTokenSource();
        // Setup a dummy Activity so Activity.Current is not null
        using var activity = new Activity("TestActivity");
        activity.Start();
        var sut = CreateSut();
        _output.WriteLine("Start retrieving data");
        await sut.RetrieveData(new StartDownloadDataEvent(), cts.Token);
        //_mockedLogger.Verify(x => x.LogDebug(It.IsAny<string>()), Times.Exactly(20));
        _mockedDaprClient.Verify(
            x => x.PublishEventAsync(
                NameConsts.INFLUX_PUBSUB_NAME,
                NameConsts.INFLUX_FINISHED_RETRIEVE_DATA,
                It.IsAny<RetrieveDataResponse>(),
                It.IsAny<CancellationToken>()
            ),
            Times.Once()
        );
        activity.Stop();
        _output.WriteLine("finished");
    }

    [Fact]
    public async Task WhenExportDataForDateHasBeenCalled_AndWeHaveData_WeUploadStuffToAzureAndDontSendData()
    {
        var sut = CreateSut();
        var dateTimeToCheck = DateTime.Now.AddDays(-1);
        var cts = new CancellationTokenSource();
        await sut.ExportDataForDate(dateTimeToCheck, cts.Token);

        _mockedDaprClient.Verify(x => x.PublishEventAsync(
            It.IsAny<string>(),
            It.IsAny<string>(),
            It.IsAny<object>(),
            It.IsAny<CancellationToken>()), Times.Never());
        _mockedFileService.Verify(x => x.UploadToAzure(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<CancellationToken>()), Times.Once());
    }

    private InfluxController CreateSut()
    {
        var expected = new List<InfluxRecord>(){
            new() {
                Humidity = 1,
                Pressure = 1, Temperature = 1, Watt = 1, Time = DateTime.Now
            }
        };
        _mockedInfluxDbService.Setup(x => x.QueryAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<CancellationToken>())).ReturnsAsync(expected);

        _mockedFileService.Setup(x => x.UploadToAzure(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);
        _localFileService.Setup(x => x.WriteToFile(It.IsAny<string>(), It.IsAny<List<InfluxRecord>?>(), It.IsAny<CancellationToken>())).Returns(Task.CompletedTask);
        _mockedFileService.Setup(x => x.RetrieveParsedFile(It.IsAny<string>(), It.IsAny<string>()))
            .ReturnsAsync("dummy-file.csv");
        _localFileService.Setup(x => x.ReadFromFile(It.IsAny<string>())).Returns(new List<InfluxRecord>{
            new InfluxRecord {
                Humidity = 1,
                Pressure = 1,
                Temperature = 1,
                Watt = 1,
                Time = DateTime.Now
            }
        });



        var cleanupService = _serviceProvider.GetRequiredService<ICleanupService>();


        var controller = new InfluxController(_mockedInfluxDbService.Object,
                        _mockedFileService.Object,
                        _configuration,
                        cleanupService,
                        _mockedDaprClient.Object,
                        _localFileService.Object,
                        _mockedLogger.Object);
        return controller;

    }
}
