using System;
using System.IO.Abstractions;
using System.Threading.Tasks;
using Common.Factory;
using Common.Services;
using DataExporter.Controllers;
using DataExporter.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit;

namespace DataExporter.Tests.ControllerTests;

public class InfluxController_RetrieveLastFileTests : IClassFixture<TestSetup>

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

    public InfluxController_RetrieveLastFileTests(TestSetup testSetup)
    {
        _serviceProvider = testSetup.ServiceProvider;
        _configuration = _serviceProvider.GetRequiredService<IConfiguration>();
    }

    [Fact]
    public async Task RetrieveLastFile_WhenFileServiceThrows_LogsErrorAndContinues()
    {
        // Arrange
        _mockedFileService.Setup(x => x.RetrieveParsedFile(It.IsAny<string>(), It.IsAny<string>()))
            .ThrowsAsync(new Exception("Test exception"));
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

        // Act
        var method = controller.GetType().GetMethod("RetrieveLastFile", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var task = (Task<string?>)method.Invoke(controller, null)!;
        var file = await task;

        // Assert
        Assert.Null(file);
        _mockedLogger.Verify(
            l => l.Log(
                LogLevel.Error,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("Test exception")),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.AtLeastOnce);
    }


    [Fact]
    public async Task RetrieveLastFile_WhenParsedFileIsNullOrEmpty_DoesNotReturnAndContinues()
    {
        // Arrange: always return null for parsedFile, so the if is never true
        _mockedFileService.Setup(x => x.RetrieveParsedFile(It.IsAny<string>(), It.IsAny<string>()))
            .ReturnsAsync((string?)null);
        // File.Exists should never be called, but if it is, return false
        _mockedFileSystem.Setup(x => x.File.Exists(It.IsAny<string>())).Returns(false);
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

        // Act
        var method = controller.GetType().GetMethod("RetrieveLastFile", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var task = (Task<string?>)method.Invoke(controller, null)!;
        var file = await task;

        // Assert
        Assert.Null(file);
        // Optionally, verify that LogWarning was called for no export file found
        _mockedLogger.Verify(
            l => l.Log(
                LogLevel.Warning,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString()!.Contains("No export file found")),
                null,
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Once);
    }
}
