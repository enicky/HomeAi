using Castle.Core.Logging;
using DataExporter.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;

namespace DataExporter.Tests.ControllerTests;

public class TestSetup
{
    public ServiceProvider ServiceProvider { get; private set; }

    public TestSetup(){
        var sc = new ServiceCollection();
        var configuration = new ConfigurationBuilder()
        .SetBasePath(Directory.GetCurrentDirectory())
        .AddJsonFile("appsettings.josn", optional: true, reloadOnChange:true)
        .Build();
        sc.AddSingleton<IConfiguration>(configuration);
        sc.AddTransient<ICleanupService, CleanupService>();
        var l = new Mock<ILogger<CleanupService>>();
        sc.AddSingleton<ILogger<CleanupService>>(l.Object);

        ServiceProvider = sc.BuildServiceProvider();
    }
}
