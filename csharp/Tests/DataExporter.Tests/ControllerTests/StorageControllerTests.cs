using Common.Helpers;
using Common.Models.AI;
using Common.Services;
using Dapr.Client;
using DataExporter.Controllers;
using DataExporter.Services;
using Meziantou.Extensions.Logging.Xunit;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit.Abstractions;

namespace DataExporter.Tests.ControllerTests;

public class StorageControllerTests : IClassFixture<TestSetup>
{
    private readonly ServiceProvider _serviceProvider;
    private readonly IConfiguration _configuration;
    private readonly ITestOutputHelper _output;
    private readonly Mock<DaprClient> _mockedDaprClient = new();
    private readonly Mock<IFileService> _mockedFileService = new();
    private readonly Mock<ILogger<StorageController>> _mockedLogger = new();
    private readonly Mock<ILocalFileService> _localFileService = new();

    public StorageControllerTests(TestSetup testSetup, ITestOutputHelper output)
    {
        _serviceProvider = testSetup.ServiceProvider;
        testSetup.ServiceCollection.AddSingleton<ILoggerProvider>(new XUnitLoggerProvider(output, true));
        _serviceProvider = testSetup.ServiceCollection.BuildServiceProvider();

        _configuration = _serviceProvider.GetRequiredService<IConfiguration>();
        _output = output;
        _output.WriteLine("Starting in constructor");
    }

    [Fact]
    public async Task WhenStartUploadingModelHasBeenCalled_AndWeGetCorrectData_WeTriggerUploadingFileToAzure()
    {
        var cts = new CancellationTokenSource();
        var sut = CreateSut();
        _output.WriteLine("Start retrieving data");
        var now = DateTime.UtcNow;
        var modelToPass = new StartUploadModel
        {
            ModelPath = "/app/checkpoints/something",
            TriggerMoment = now,
        };
        await sut.StartUploadingModelToAzure(modelToPass, _mockedDaprClient.Object, cts.Token);
        _mockedFileService.Verify(x => x.UploadToAzure(StorageHelpers.ModelContainerName, It.IsAny<string>(), It.Is<CancellationToken>(x => x == cts.Token)));

        //verify that fileservice has uploaded to azure
        _output.WriteLine("finished");
    }

    [Fact]
    public async Task WhenStartUploadingModelHasBeenCalled_AndWeGetIncorrectData_WeDoNotTriggerUploadingFileToAzure()
    {
        var cts = new CancellationTokenSource();
        var sut = CreateSut();
        _output.WriteLine("Start retrieving data");
        var now = DateTime.UtcNow;
        var modelToPass = new StartUploadModel
        {
            ModelPath = "/app/cc/something",
            TriggerMoment = now,
        };
        await sut.StartUploadingModelToAzure(modelToPass, _mockedDaprClient.Object, cts.Token);
        _mockedFileService.Verify(x => x.UploadToAzure(StorageHelpers.ModelContainerName, It.IsAny<string>(), It.Is<CancellationToken>(x => x == cts.Token)), Times.Never());

        //verify that fileservice has uploaded to azure
        _output.WriteLine("finished");
    }

    private StorageController CreateSut()
    {
        _mockedFileService
            .Setup(x =>
                x.UploadToAzure(
                    StorageHelpers.ContainerName,
                    It.IsAny<string>(),
                    It.IsAny<CancellationToken>()
                )
            )
            .Returns(Task.CompletedTask);
        _localFileService.Setup(x => x.CopyFile(It.IsAny<string>(), It.IsAny<string>())).Returns(Task.CompletedTask);
        ILogger<StorageController> logger = XUnitLogger.CreateLogger<StorageController>(_output);

        var controller = new StorageController(logger, _mockedFileService.Object, _localFileService.Object);

        return controller;
    }
}
