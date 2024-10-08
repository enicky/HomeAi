using System.Net;
using app.Services;
using Common.Helpers;
using Common.Models;
using Common.Models.Influx;
using Common.Models.Responses;
using Common.Services;
using Dapr.Client;
using DataExporter.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit.Abstractions;

namespace DataExporter.Tests.ControllerTests;

public class InfluxControllerTests : IClassFixture<TestSetup>
{
    private readonly ServiceProvider _serviceProvider;
    private readonly IConfiguration _configuration;
    private readonly ITestOutputHelper _output;
    private readonly Mock<IInfluxDbService> _mockedInfluxDbService = new();
    private readonly Mock<IFileService> _mockedFileService = new();
    private readonly Mock<DaprClient> _mockedDaprClient = new();
    private readonly Mock<ILocalFileService> _localFileService = new();
    private readonly Mock<ILogger<InfluxController.Controllers.InfluxController>> _mockedLogger = new();

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
        var sut = CreateSut();
        _output.WriteLine("Start retrieving data");
        await sut.RetrieveData(cts.Token);
        //_mockedLogger.Verify(x => x.LogDebug(It.IsAny<string>()), Times.Exactly(20));
        _mockedDaprClient.Verify(x =>
            x.PublishEventAsync(NameConsts.INFLUX_PUBSUB_NAME, NameConsts.INFLUX_FINISHED_RETRIEVE_DATA, It.IsAny<RetrieveDataResponse>(), It.IsAny<CancellationToken>()), Times.Once());

        _output.WriteLine("finished");
    }

    [Fact]
    public async Task WhenExportDataForDateHasBeenCalled_AndWeHaveData_WeUploadStuffToAzureAndDontSendData()
    {
        var sut = CreateSut();
        var dateTimeToCheck = DateTime.Now.AddDays(-1);
        var cts = new CancellationTokenSource();
        await sut.ExportDataForDate(dateTimeToCheck, cts.Token);

        _mockedDaprClient.Verify(x => x.PublishEventAsync(It.IsAny<string>(), It.IsAny<string>(), cts.Token), Times.Never());
        _mockedFileService.Verify(x => x.UploadToAzure(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<CancellationToken>()), Times.Once());
    }

    private InfluxController.Controllers.InfluxController CreateSut()
    {
        var expected = new List<InfluxRecord>(){
            new() {
                Humidity = 1,
                Pressure = 1, Temperature = 1, Watt = 1, Time = DateTime.Now
            }
        };
        _mockedInfluxDbService.Setup(x => x.QueryAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<CancellationToken>())).ReturnsAsync(expected);

        _mockedFileService.Setup(x => x.UploadToAzure(StorageHelpers.ContainerName, It.IsAny<string>(), It.IsAny<CancellationToken>())).Returns(Task.CompletedTask);
        _localFileService.Setup(x => x.WriteToFile(It.IsAny<string>(), It.IsAny<List<InfluxRecord>?>(), It.IsAny<CancellationToken>())).Returns(Task.CompletedTask);



        var cleanupService = _serviceProvider.GetRequiredService<ICleanupService>();


        var controller = new InfluxController.Controllers.InfluxController(_mockedInfluxDbService.Object,
                        _mockedFileService.Object,
                        _configuration,
                        cleanupService,
                        _mockedDaprClient.Object,
                        _localFileService.Object,
                        _mockedLogger.Object);
        return controller;

    }
}
