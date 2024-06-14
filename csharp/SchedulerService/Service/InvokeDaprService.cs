
using Common.Models.Responses;
using Dapr.Client;


namespace SchedulerService.Service;

public interface IInvokeDaprService{
    Task TriggerExportData(CancellationToken token = default);
    Task TriggerTrainingOfAiModel(CancellationToken token = default);
}

public class InvokeDaprService : IInvokeDaprService{
    private readonly ILogger<InvokeDaprService> _logger;
    public InvokeDaprService(ILogger<InvokeDaprService> logger){
        _logger = logger;
    }

    public async Task TriggerExportData(CancellationToken token = default){
        const string logPrefix = "[InvokeDaprService:TriggerExportData]";
         using var client = new DaprClientBuilder().Build();
         var retrieveResponse = await client.InvokeMethodAsync<RetrieveDataResponse>(HttpMethod.Get, "dataexporter", "api/Influx/retrievedata",  token);
         if(retrieveResponse.Success){
            _logger.LogInformation($"{logPrefix} Successfully triggered retrieving data from Influx");
            return;
         }
         _logger.LogWarning($"{logPrefix} There was an issue retrieving data from Influx");         
    }

    public async Task TriggerTrainingOfAiModel(CancellationToken token = default)
    {
        const string logPrefix= "[InvokeDaprService:TriggerTrainingOfAiModel]";
        using var client = new DaprClientBuilder().Build();
        var retrieveResponse = await client.InvokeMethodAsync<TrainAiModelResponse>(HttpMethod.Get, "modeltrainer", "/train_model", token);
        if(retrieveResponse.Success){
            _logger.LogInformation($"{logPrefix} Successfully triggered training of AI Model");
            return;
        }
        _logger.LogWarning($"{logPrefix} There was an issue triggering the training of the AI model");
    }
}