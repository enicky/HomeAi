using DataExporter.Services.Factory;

namespace DataExporter.Tests.Wrappers;

public class InfluxDbClientFactoryTests
{
    [Fact]
    public void CanCreateWrapperFromFactory(){
        var factory = new InfluxDbClientFactory();
        var result = factory.CreateWrapper("a", "b");
        Assert.NotNull(result);
    }
    
}
