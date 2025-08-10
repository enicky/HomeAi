
using Hangfire;
using SchedulerService.Service;
using SchedulerService.Triggers;
using Microsoft.ApplicationInsights.Extensibility;
using Common.ApplicationInsights.Filter;
using Common.ApplicationInsights.Initializers;
using Microsoft.ApplicationInsights.DependencyCollector;

namespace SchedulerService;

public class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);
        builder.Configuration.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);
#if DEBUG
        builder.Configuration.AddJsonFile(
            "appsettings.Development.json",
            optional: true,
            reloadOnChange: true
        );
#endif
        builder.Configuration.AddEnvironmentVariables();
        builder.Configuration.AddCommandLine(args);
        // builder.Configuration.AddJsonFile(
        //     "appsettings.json",
        //     optional: false,
        //     reloadOnChange: true
        // );
        // builder.Configuration.AddEnvironmentVariables();
        // builder.Configuration.AddCommandLine(args);

        builder.Services.AddApplicationInsightsTelemetry();
        builder.Services.AddApplicationInsightsTelemetryProcessor<SqlDependencyFilter>();
        builder.Services.AddApplicationInsightsTelemetryProcessor<HangfireRequestFilter>();
        builder.Services.AddApplicationInsightsTelemetryProcessor<HealthzRequestFilter>();
        builder.Services.AddSingleton<ITelemetryInitializer>(x => new CustomTelemetryInitializer(
            "SchedulerService"
        ));
        builder.Services.ConfigureTelemetryModule<DependencyTrackingTelemetryModule>((module, o) => module.EnableSqlCommandTextInstrumentation = true);
        
        builder.Services.AddHealthChecks();
        builder.Services.AddDaprClient();
        builder.Services.AddControllers().AddDapr();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();
        builder.Services.AddLogging(loggingBuilder =>
        {
            loggingBuilder.ClearProviders();
            loggingBuilder.AddSimpleConsole(options =>
            {
                options.IncludeScopes = false;
                options.SingleLine = true;
                options.TimestampFormat = "HH:mm:ss ";
            });
        });

        builder.Logging.AddApplicationInsights(
            configureTelemetryConfiguration: (config) =>
                config.ConnectionString = builder.Configuration.GetValue<string>(
                    "ApplicationInsights:ConnectionString"
                ),
            configureApplicationInsightsLoggerOptions: (options) => { }
        );

        string sql_config = builder.Configuration.GetValue<string>("SQL_CONFIG")!;
        string sql_USERNAME = builder.Configuration.GetValue<string>("SQL_USERNAME")!;
        string sql_PASSWORD = builder.Configuration.GetValue<string>("SQL_PASSWORD")!;
        string sql_SERVER = builder.Configuration.GetValue<string>("SQL_SERVER")!;
        sql_config = sql_config
            .Replace("[SQL_USERNAME]", sql_USERNAME)
            .Replace("[SQL_PASSWORD]", sql_PASSWORD)
            .Replace("[SQL_SERVER]", sql_SERVER);

        string schedule = builder.Configuration.GetValue<string>("SCHEDULE")!;
        string schedule_train_model = builder.Configuration.GetValue<string>(
            "SCHEDULE_TRAIN_MODEL"
        )!;
        bool enable_extra_train_model = bool.Parse(
            builder.Configuration.GetValue("ENABLE_EXTRA_MODEL_TRAIN", "false")!
        );

        if (string.IsNullOrEmpty(schedule))
        {
            schedule = "* 8 * * *"; // every day at 8 AM
        }
        if (string.IsNullOrEmpty(schedule_train_model))
            schedule_train_model = "* 10 * * *"; // every day at 10 AM => Way to fast. But hey ...

        Console.WriteLine($"using sql config : {sql_config}");
        Console.WriteLine($"Using schedule {schedule}");

        builder.Services.AddHangfire(configuration =>
            configuration
                .SetDataCompatibilityLevel(CompatibilityLevel.Version_180)
                .UseSimpleAssemblyNameTypeSerializer()
                .UseRecommendedSerializerSettings()
                .UseSqlServerStorage(sql_config)
        );
        builder.Services.AddHangfireServer();
        builder.Services.AddScoped<IInvokeDaprService, InvokeDaprService>();

        
        var app = builder.Build();

        app.UseCloudEvents();
        app.UseSwagger();
        app.UseSwaggerUI();
        

        app.MapControllers();
        app.MapSubscribeHandler();
        
        app.UseHangfireDashboard();
        RecurringJob.AddOrUpdate<TriggerRetrieveDataForAi>(
            "trigger_ai_job",
            x => x.RunAsync(default),
            schedule,
            new RecurringJobOptions { TimeZone = TimeZoneInfo.Local }
        );
        if (enable_extra_train_model)
        {
            RecurringJob.AddOrUpdate<TriggerTrainAiModel>(
                "trigger_train_model",
                x => x.RunAsync(default),
                schedule_train_model,
                new RecurringJobOptions { TimeZone = TimeZoneInfo.Local }
            );
        }

        app.MapHealthChecks("/healthz");
        app.UseAuthorization();
        app.Run();
    }
}
