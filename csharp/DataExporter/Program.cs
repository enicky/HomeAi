using System.Diagnostics.CodeAnalysis;
using app.Services;
using Common.Services;
using DataExporter.Extensions;
using DataExporter.Services;

namespace DataExporter;
[ExcludeFromCodeCoverage]
internal static class Program
{
    private static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);
        builder.Services.AddHealthChecks(); // enable /healtz
        builder.Configuration.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);
        #if DEBUG
        builder.Configuration.AddJsonFile("appsettings.Development.json", optional: false, reloadOnChange: true);
        #endif
        builder.Configuration.AddEnvironmentVariables();
        builder.Configuration.AddCommandLine(args);

        builder.Services.AddLogging(loggingBuilder =>{
            loggingBuilder.ClearProviders();
            loggingBuilder.AddSimpleConsole(options => {
                options.IncludeScopes = false;
                options.SingleLine = true;
                options.TimestampFormat = "HH:mm:ss ";
            });
        });
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

