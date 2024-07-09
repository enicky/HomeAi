using app.Services;

internal class Program
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
                options.IncludeScopes = true;
                options.TimestampFormat = "HH:mm:ss";
            });
        });
        // Add services to the container.
        // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
        builder.Services.AddControllers().AddDapr();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();
        builder.Services.AddSingleton<InfluxDBService>();
        builder.Services.AddScoped<IFileService, FileService>();

        var app = builder.Build();

        app.UseSwagger();
        app.UseSwaggerUI();

        app.MapSubscribeHandler();
        app.MapControllers();

        app.Run();
    }
}

record WeatherForecast(DateOnly Date, int TemperatureC, string? Summary)
{
    public int TemperatureF => 32 + (int)(TemperatureC / 0.5556);
}
