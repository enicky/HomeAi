using System.Net;
using app.Services;
using Azure;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using Common.Exceptions;
using Common.Factory;
using Common.Services;
using DataExporter.Services.Factory;
using DataExporter.Tests.ControllerTests;
using Meziantou.Extensions.Logging.Xunit;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit.Abstractions;

namespace DataExporter.Tests.Services;

public class FileServiceTests : IClassFixture<TestSetup>
{
    private readonly ITestOutputHelper _output;
    private readonly TestSetup _testSetup;
    private readonly Mock<IBlobServiceClientFactory> _mockBlobServiceClientFactory;
    private readonly Mock<BlobServiceClient> _mockBlobServiceClient;
    private readonly Mock<BlobContainerClient> _mockBlobContainerClient;
    private readonly Mock<BlobClient> _mockBlobClient;
    private readonly Mock<ILogger<FileService>> _logger = new();
    private readonly IServiceProvider _serviceProvider;
    private readonly IConfiguration _configuration;

    public FileServiceTests(TestSetup testSetup, ITestOutputHelper testOutputHelper)
    {
        _output = testOutputHelper;
        _serviceProvider = testSetup.ServiceProvider;
        testSetup.ServiceCollection.AddSingleton<ILoggerProvider>(new XUnitLoggerProvider(testOutputHelper, true));
        _serviceProvider = testSetup.ServiceCollection.BuildServiceProvider();

        _configuration = _serviceProvider.GetRequiredService<IConfiguration>();
        _testSetup = testSetup;

        _mockBlobServiceClientFactory = new Mock<IBlobServiceClientFactory>();
        _mockBlobServiceClient = new Mock<BlobServiceClient>();
        _mockBlobContainerClient = new Mock<BlobContainerClient>();
        _mockBlobClient = new Mock<BlobClient>();
    }

    [Fact]
    public async Task UploadToAzure_AndContainerExists_ShouldStoreFileInAzure()
    {
        var cts = new CancellationTokenSource();
        _mockBlobServiceClientFactory.Setup(f => f.Create(It.IsAny<string>(), It.IsAny<string>()))
                           .Returns(_mockBlobServiceClient.Object);

        _mockBlobServiceClient.Setup(x => x.GetBlobContainerClient(It.IsAny<string>()))
                    .Returns(_mockBlobContainerClient.Object);

        _mockBlobContainerClient.Setup(x => x.GetBlobClient(It.IsAny<string>()))
            .Returns(_mockBlobClient.Object);
        _mockBlobContainerClient.Setup(x => x.Name).Returns("testContainer");

        var mockResponse = new Mock<Response<bool>>();
        mockResponse.SetupGet(x => x.Value).Returns(true);


        _mockBlobContainerClient.Setup(x => x.ExistsAsync(cts.Token))
            .Returns(Task.FromResult(mockResponse.Object));

        _mockBlobClient
            .Setup(x => x.UploadAsync("test.txt", true, cts.Token))
            .Returns(Task.FromResult(Mock.Of<Response<BlobContentInfo>>()));
        var _logger = XUnitLogger.CreateLogger<FileService>(_output);

        var sut = new FileService(_configuration, _mockBlobServiceClientFactory.Object, _logger);
        var containerName = "testContainer";
        var generatedFileName = "test.txt";

        await sut.UploadToAzure(containerName, generatedFileName, cts.Token);

        _mockBlobContainerClient.Verify(x => x.CreateIfNotExistsAsync(PublicAccessType.None, null, null, cts.Token), Times.Once());
        _mockBlobContainerClient.Verify(x => x.ExistsAsync(cts.Token), Times.Once());
        mockResponse.Verify(x => x.Value, Times.Once());

        _mockBlobClient.Verify(x => x.UploadAsync(generatedFileName, true, cts.Token), Times.Once());

    }

    [Fact]
    public async Task UploadToAzure_AndContainerDoesNotExist_ShouldCreateContainerAndShouldStoreFileInAzure()
    {
        var cts = new CancellationTokenSource();
        _mockBlobServiceClientFactory.Setup(f => f.Create(It.IsAny<string>(), It.IsAny<string>()))
                           .Returns(_mockBlobServiceClient.Object);

        _mockBlobServiceClient.Setup(x => x.GetBlobContainerClient(It.IsAny<string>()))
                    .Returns(_mockBlobContainerClient.Object);

        _mockBlobContainerClient.Setup(x => x.GetBlobClient(It.IsAny<string>()))
            .Returns(_mockBlobClient.Object);
        _mockBlobContainerClient.Setup(x => x.Name).Returns("testContainer");

        var mockResponse = new Mock<Response<bool>>();
        mockResponse.SetupGet(x => x.Value).Returns(false);


        _mockBlobContainerClient.Setup(x => x.ExistsAsync(cts.Token))
            .Returns(Task.FromResult(mockResponse.Object));

        _mockBlobClient
            .Setup(x => x.UploadAsync("test.txt", true, cts.Token))
            .Returns(Task.FromResult(Mock.Of<Response<BlobContentInfo>>()));
        var _logger = XUnitLogger.CreateLogger<FileService>(_output);

        var sut = new FileService(_configuration, _mockBlobServiceClientFactory.Object, _logger);
        var containerName = "testContainer";
        var generatedFileName = "test.txt";

        await sut.UploadToAzure(containerName, generatedFileName, cts.Token);

        _mockBlobContainerClient.Verify(x => x.CreateIfNotExistsAsync(PublicAccessType.None, null, null, cts.Token), Times.Once());
        _mockBlobContainerClient.Verify(x => x.ExistsAsync(cts.Token), Times.Once());
        mockResponse.Verify(x => x.Value, Times.Once());

        _mockBlobClient.Verify(x => x.UploadAsync(generatedFileName, true, cts.Token), Times.Once());

    }

    [Fact]
    public void IfAccountNameIsEmpty_ShouldThrowAnException()
    {
        var _logger = XUnitLogger.CreateLogger<FileService>(_output);


        var emptyConfiguration = new ConfigurationBuilder().Build();

        Assert.Throws<AccountNameNullException>(() =>
        {
            new FileService(emptyConfiguration, _mockBlobServiceClientFactory.Object, _logger);
        });

    }

    [Fact]
    public void IfAccountKeyIsEmpty_ShouldThrowAnException()
    {
        var _logger = XUnitLogger.CreateLogger<FileService>(_output);
        IEnumerable<KeyValuePair<string, string?>> inMemorySettings = new List<KeyValuePair<string, string?>>{
            new("accountName", "test")
        };
        var emptyConfiguration = new ConfigurationBuilder().AddInMemoryCollection(inMemorySettings).Build();
        Assert.Throws<AccountKeyNullException>(() =>
        {
            new FileService(emptyConfiguration, _mockBlobServiceClientFactory.Object, _logger);
        });
    }

    [Fact]
    public async Task WhenStorageCallsThrowAnException_AndExceptionGetsCaught_PerformRestOfCode()
    {
        var cts = new CancellationTokenSource();
        _mockBlobServiceClientFactory.Setup(f => f.Create(It.IsAny<string>(), It.IsAny<string>()))
                           .Returns(_mockBlobServiceClient.Object);

        _mockBlobServiceClient.Setup(x => x.GetBlobContainerClient(It.IsAny<string>()))
                    .Returns(_mockBlobContainerClient.Object);

        _mockBlobContainerClient.Setup(x => x.GetBlobClient(It.IsAny<string>()))
            .Returns(_mockBlobClient.Object);
        _mockBlobContainerClient.Setup(x => x.Name).Returns("testContainer");

        var mockResponse = new Mock<Response<bool>>();
        mockResponse.SetupGet(x => x.Value).Returns(false);

        _mockBlobContainerClient.Setup(x => x.CreateIfNotExistsAsync(PublicAccessType.None, null, null, cts.Token))
            .Throws(new RequestFailedException("test"));

        _mockBlobContainerClient.Setup(x => x.ExistsAsync(cts.Token))
            .Returns(Task.FromResult(mockResponse.Object));

        _mockBlobClient
            .Setup(x => x.UploadAsync("test.txt", true, cts.Token))
            .Returns(Task.FromResult(Mock.Of<Response<BlobContentInfo>>()));
        var _logger = XUnitLogger.CreateLogger<FileService>(_output);

        var sut = new FileService(_configuration, _mockBlobServiceClientFactory.Object, _logger);
        var containerName = "testContainer";
        var generatedFileName = "test.txt";

        await sut.UploadToAzure(containerName, generatedFileName, cts.Token);


        _mockBlobContainerClient.Verify(x => x.CreateIfNotExistsAsync(PublicAccessType.None, null, null, cts.Token), Times.Once());
        _mockBlobContainerClient.Verify(x => x.ExistsAsync(cts.Token), Times.Never());
        mockResponse.Verify(x => x.Value, Times.Never());

        _mockBlobClient.Verify(x => x.UploadAsync(generatedFileName, true, cts.Token), Times.Once());

    }

    [Fact]
    public async Task DownloadFromAzureShouldWork()
    {
        var cts = new CancellationTokenSource();
        _mockBlobServiceClientFactory.Setup(f => f.Create(It.IsAny<string>(), It.IsAny<string>()))
                           .Returns(_mockBlobServiceClient.Object);

        _mockBlobServiceClient.Setup(x => x.GetBlobContainerClient(It.IsAny<string>()))
                    .Returns(_mockBlobContainerClient.Object);
        _mockBlobContainerClient.Setup(x => x.GetBlobClient(It.IsAny<string>()))
                    .Returns(_mockBlobClient.Object);

        _mockBlobClient.Setup(x => x.DownloadToAsync(It.IsAny<string>())).Returns(Task.FromResult(Mock.Of<Response>()));
        var _logger = XUnitLogger.CreateLogger<FileService>(_output);

        var sut = new FileService(_configuration, _mockBlobServiceClientFactory.Object, _logger);
        var exception = await Record.ExceptionAsync(() => sut.RetrieveParsedFile("test", "test"));
        Assert.Null(exception);
    }
}
