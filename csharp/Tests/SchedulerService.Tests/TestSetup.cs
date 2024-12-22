using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace SchedulerService.Tests;

public class TestSetup
{
    public ServiceProvider ServiceProvider { get; private set; }
    public ServiceCollection ServiceCollection { get; private set; } = new();
    public IConfiguration Configuration { get; private set; }

    public TestSetup(){
        var configuration = new ConfigurationBuilder()
        .SetBasePath(Directory.GetCurrentDirectory())
        .AddJsonFile("appsettings.json", optional: true, reloadOnChange:true)
        .AddJsonFile("appsettings.Development.json", optional: true, reloadOnChange:true)
        .Build();
        Configuration = configuration;
        ServiceCollection.AddSingleton<IConfiguration>(configuration);
        //ServiceCollection.AddTransient<ICleanupService, CleanupService>();
        //var l = new Mock<ILogger<CleanupService>>();
        //ServiceCollection.AddSingleton<ILogger<CleanupService>>(l.Object);
        

        ServiceProvider = ServiceCollection.BuildServiceProvider();
    }
}
