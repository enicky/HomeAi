using app.Services;
using DataExporter.Tests.ControllerTests;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Xunit.Abstractions;

namespace DataExporter.Tests.Services;

public class FileServiceTests
{
    private readonly ITestOutputHelper _output;
    private readonly TestSetup _testSetup;
    private readonly IServiceProvider _serviceProvider;
    private readonly IConfiguration _configuration;

    public FileServiceTests(TestSetup testSetup, ITestOutputHelper testOutputHelper){
        _output = testOutputHelper;
        _serviceProvider = testSetup.ServiceProvider;
        _configuration = _serviceProvider.GetRequiredService<IConfiguration>();
        _testSetup = testSetup;
    }

    [Fact]
    public async Task UploadToAzure_ShouldStoreFileInAzure(){
        var sut = new FileService(_configuration);
        var containerName = "testContainer";
        var generatedFileName = "test.txt";
        
        await sut.UploadToAzure(containerName, generatedFileName);

        Assert.True(false);

    }
}
