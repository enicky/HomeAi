using System.Diagnostics.CodeAnalysis;
using app.Services;
using Common.ApplicationInsights.Filter;
using Common.Services;
using DataExporter.Extensions;
using DataExporter.Services;
using Microsoft.ApplicationInsights.DependencyCollector;

namespace DataExporter;
[ExcludeFromCodeCoverage]
internal static class Program
{
    private static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);
        builder.Configuration.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);
        #if DEBUG
        builder.Configuration.AddJsonFile("appsettings.Development.json", optional: false, reloadOnChange: true);
        #endif
        builder.Configuration.AddEnvironmentVariables();
        builder.Configuration.AddCommandLine(args);

        builder.Services.AddApplicationInsightsTelemetry(); // add Application Insights
        builder.Services.AddApplicationInsightsTelemetryProcessor<HealthzRequestFilter>();
        
        builder.Services.ConfigureTelemetryModule<DependencyTrackingTelemetryModule>((module, o) => module.EnableSqlCommandTextInstrumentation = true);
        
        builder.Services.AddHealthChecks(); // enable /healtz
        
        builder.Services.AddLogging(loggingBuilder =>{
            loggingBuilder.ClearProviders();
            loggingBuilder.AddSimpleConsole(options => {
                options.IncludeScopes = false;
                options.SingleLine = true;
                options.TimestampFormat = "HH:mm:ss ";
            });
        });
        builder.Logging.AddApplicationInsights(configureTelemetryConfiguration: (config) => 
            config.ConnectionString = builder.Configuration.GetValue<string>("ApplicationInsights:ConnectionString"),
            configureApplicationInsightsLoggerOptions: (options) => { }
        );
        builder.Services.AddDaprClient();
        // Add services to the container.
        // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
        builder.Services.AddControllers().AddDapr();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();
        builder.Services.AddServices();

        var app = builder.Build();
        app.UseCloudEvents();
        app.UseSwagger();
        app.UseSwaggerUI();


        app.MapControllers();
        app.MapSubscribeHandler();

        app.Run();
    }
}

