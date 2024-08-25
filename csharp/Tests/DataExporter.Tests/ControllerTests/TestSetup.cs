using Castle.Core.Logging;
using DataExporter.Services;
using Meziantou.Extensions.Logging.Xunit;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit.Abstractions;

namespace DataExporter.Tests.ControllerTests;

public class    TestSetup
{
    public ServiceProvider ServiceProvider { get; private set; }
    public ServiceCollection ServiceCollection { get; private set; }

    public TestSetup(){
        ServiceCollection = new ServiceCollection();
        var configuration = new ConfigurationBuilder()
        .SetBasePath(Directory.GetCurrentDirectory())
        .AddJsonFile("appsettings.json", optional: true, reloadOnChange:true)
        .Build();
        ServiceCollection.AddSingleton<IConfiguration>(configuration);
        ServiceCollection.AddTransient<ICleanupService, CleanupService>();
        var l = new Mock<ILogger<CleanupService>>();
        ServiceCollection.AddSingleton<ILogger<CleanupService>>(l.Object);

        ServiceProvider = ServiceCollection.BuildServiceProvider();
    }
}
