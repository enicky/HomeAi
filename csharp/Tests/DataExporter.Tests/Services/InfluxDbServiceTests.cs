using Common.Models.Influx;
using DataExporter.Services;
using DataExporter.Services.Factory;
using DataExporter.Services.Wrapper;
using DataExporter.Tests.ControllerTests;
using Meziantou.Extensions.Logging.Xunit;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Moq;
using Xunit.Abstractions;

namespace DataExporter.Tests.Services;

public class InfluxDbServiceTests : IClassFixture<TestSetup>
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ITestOutputHelper _output;
    private readonly IConfiguration _configuration;

    public InfluxDbServiceTests(TestSetup testSetup, ITestOutputHelper testOutputHelper)
    {
        _output = testOutputHelper;
        _serviceProvider = testSetup.ServiceProvider;
        _configuration = _serviceProvider.GetRequiredService<IConfiguration>();
    }

    private readonly Mock<IInfluxDbClientFactory> _mockInfluxDbClientFactory = new();
    private readonly Mock<IInfluxDbClientWrapper> _mockInfluxDbClientWrapper = new();

    [Fact]
    public void InfluxRecord_CanBeCreated()
    {
        var record = new InfluxRecord();
        Assert.NotNull(record);
    }

    // test
    [Fact]
    public async Task Test()
    {
        var dataToReturn = new List<InfluxRecord>{
        new InfluxRecord{Humidity = 1, Pressure = 1, Temperature = 1, Time = DateTime.Today, Watt = 10}
       };

        var cts = new CancellationTokenSource();
        string queryString = "import \"experimental\"" +
           " from(bucket: \"home_assistant\")" +
           " |> range(start:experimental.subDuration(d: 24h, from: today()), stop: today())" +
           " |> filter(fn: (r) => r[\"entity_id\"] == \"forecast_home_2\" or r[\"entity_id\"] == \"warmtepomp_power\" or r[\"entity_id\"] == \"smoke_detector_device_17_temperature\")" +
           " |> filter(fn: (r) =>  r[\"_field\"] == \"humidity\" or r[\"_field\"] == \"pressure\" or r[\"_field\"] == \"value\")" +
           " |> aggregateWindow(every: 1m, fn: mean, createEmpty: true)" +
           " |> fill(usePrevious: true)" +
           " |> keep(columns: [\"_field\", \"_measurement\", \"_value\", \"_time\"])" +
           " |> pivot(" +
           "     rowKey: [\"_time\"], columnKey: [\"_measurement\", \"_field\"], valueColumn: \"_value\")" +

           " |> yield(name: \"values\")";
        var orgString = "e6601eb7b60be0fe";


        _mockInfluxDbClientFactory.Setup(f => f.CreateWrapper(It.IsAny<string>(), It.IsAny<string>()))
                           .Returns(_mockInfluxDbClientWrapper.Object);
        _mockInfluxDbClientWrapper.Setup(f => f.GetData(It.IsAny<string>(), orgString, cts.Token)).Returns(Task.FromResult(dataToReturn));

        var _logger = XUnitLogger.CreateLogger<InfluxDbService>(_output);
        var sut = new InfluxDbService(_configuration, _mockInfluxDbClientFactory.Object, _logger);
        var result = await sut.QueryAsync(queryString, orgString, cts.Token);
        Assert.NotNull(result);
        Assert.NotEmpty(result);
        Assert.Single(result);
        Assert.Equal(1, result.First().Humidity);
        Assert.Equal(10, result.First().Watt);
    }
}
