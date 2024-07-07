using Hangfire;
using SchedulerService.Service;
using SchedulerService.Triggers;

namespace SchedulerService;

public class Program
{


    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);
        builder.Services.AddHealthChecks();
        builder.Configuration.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);
        builder.Configuration.AddEnvironmentVariables();
        builder.Configuration.AddCommandLine(args);
        builder.Services.AddControllers().AddDapr();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();

        

        string sql_config = builder.Configuration.GetValue<string>("SQL_CONFIG")!;
        string sql_USERNAME = builder.Configuration.GetValue<string>("SQL_USERNAME")!;
        string sql_PASSWORD = builder.Configuration.GetValue<string>("SQL_PASSWORD")!;
        string sql_SERVER = builder.Configuration.GetValue<string>("SQL_SERVER")!;
        sql_config = sql_config.Replace("[SQL_USERNAME]", sql_USERNAME)
                                .Replace("[SQL_PASSWORD]", sql_PASSWORD)
                                .Replace("[SQL_SERVER]", sql_SERVER);
        
        string schedule = builder.Configuration.GetValue<string>("SCHEDULE")!;
        string schedule_train_model = builder.Configuration.GetValue<string>("SCHEDULE_TRAIN_MODEL")!;

        if(string.IsNullOrEmpty(schedule)){
            schedule = "* 8 * * *"; // every day at 8 AM
        }
        if(string.IsNullOrEmpty(schedule_train_model)) schedule_train_model = "* 10 * * *"; // every day at 10 AM => Way to fast. But hey ...

        Console.WriteLine($"using sql config : {sql_config}");
        Console.WriteLine($"Using schedule {schedule}");

        // Add services to the container.
        //builder.Services.AddAuthorization();
        builder.Services.AddHangfire(configuration => configuration
        .SetDataCompatibilityLevel(CompatibilityLevel.Version_180)
            .UseSimpleAssemblyNameTypeSerializer()
            .UseRecommendedSerializerSettings()
            .UseSqlServerStorage(sql_config));
        builder.Services.AddHangfireServer();
        builder.Services.AddScoped<IInvokeDaprService, InvokeDaprService>();

        // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
        //builder.Services.AddEndpointsApiExplorer();
        //builder.Services.AddSwaggerGen();

        var app = builder.Build();

        // Configure the HTTP request pipeline.
        //if (app.Environment.IsDevelopment())
        //{
            app.UseSwagger();
            app.UseSwaggerUI();
        //}

        app.MapSubscribeHandler();
        app.MapControllers();
        var cts = new CancellationTokenSource();

        app.UseHangfireDashboard();
        RecurringJob.AddOrUpdate<TriggerRetrieveDataForAi>("trigger_ai_job", x => x.RunAsync(cts.Token), schedule, new RecurringJobOptions{TimeZone = TimeZoneInfo.Local});
        RecurringJob.AddOrUpdate<TriggerTrainAiModel>("trigger_train_model", x => x.RunAsync(cts.Token),schedule_train_model, new RecurringJobOptions{TimeZone = TimeZoneInfo.Local} );

        app.MapHealthChecks("/healthz");
        app.UseAuthorization();
        app.Run();
    }
}
