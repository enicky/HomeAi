using System.Diagnostics.CodeAnalysis;
using app.Services;
using Common.Factory;
using Common.Services;
using DataExporter.Services;
using DataExporter.Services.Factory;

namespace DataExporter.Extensions;

[ExcludeFromCodeCoverage]
public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddServices(this IServiceCollection services)
    {
        return services.AddTransient<IInfluxDbClientFactory, InfluxDbClientFactory>()
            .AddSingleton<IInfluxDbService, InfluxDBService>()
            .AddScoped<IFileService, FileService>()
            .AddTransient<ICleanupService, CleanupService>()
            .AddTransient<ILocalFileService, LocalFileService>()
            .AddTransient<IBlobServiceClientFactory, BlobServiceClientFactory>();
    }
}
