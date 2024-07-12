using System.Threading.Tasks.Dataflow;
using Common.Helpers;
using Common.Models;
using Common.Models.Responses;
using Dapr;
using Dapr.Client;
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;

namespace SchedulerService.Controllers;

[Route("api/[controller]")]
[ApiController]
public class DaprController : ControllerBase
{
    private readonly ILogger<DaprController> logger;

    public DaprController(ILogger<DaprController> logger)
    {
        this.logger = logger;
    }

    [Topic(NameConsts.INFLUX_PUBSUB_NAME, "testreply")]
    [HttpPost("testreply")]
    public IActionResult TestReply([FromBody] OrderReceived o)
    {
        logger.LogInformation($"TestReply got triggered");
        if(o is not null){
            logger.LogInformation($"Received OrderReceived response : {o.Success} for id {o.OrderId}");
            return Ok();
        }
        return BadRequest();
    }


    [Topic(pubsubName: NameConsts.INFLUX_PUBSUB_NAME, name: NameConsts.INFLUX_FINISHED_RETRIEVE_DATA)]
    [HttpPost("DownloadDataHasFinished")]
    public async Task DownloadDataHasFinished(CloudEvent<RetrieveDataResponse> response, CancellationToken token = default)
    {
        logger.LogInformation($"Response value : {JsonConvert.SerializeObject(response)}");
        logger.LogInformation($"Retrieved info that download has been completed {response?.Data?.Success} in filename {response?.Data?.GeneratedFileName}");
        logger.LogInformation($"Start Training ai model ? {response?.Data?.StartTrainingModel}");
        var mustTrainModel = false;
        if (response!.Data != null && response.Data.StartTrainingModel != null && response.Data.StartTrainingModel == true)
        { mustTrainModel = true; }

        if (mustTrainModel)
        {
            logger.LogInformation("Start triggering of downloading data to python training container");

            using var client = new DaprClientBuilder().Build();
            await client.PublishEventAsync(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_START_DOWNLOAD_DATA, token);
            logger.LogInformation("Sent message to AI container to start Downloading data and prepare it to start training model");
        }
        logger.LogInformation($"Finished exchanging messages");
    }

    [Topic(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_FINISHED_DOWNLOAD_DATA)]
    [HttpPost("AiDownloadFinishedStartTraining")]
    public async Task AiDownloadFinishedStartTraining(CancellationToken token = default)
    {
        logger.LogInformation("Retrieved info that download of data has been finished");
        logger.LogInformation("Send message to start training model on AI container");
        using var client = new DaprClientBuilder().Build();
        await client.PublishEventAsync(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_START_TRAIN_MODEL);
        logger.LogInformation("Finished sending message to AI container to start training model");
    }

    [Topic(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_FINISHED_TRAIN_MODEL)]
    [HttpPost("AiFinishedTrainingModel")]
    public Task AiFinishedTrainingModel(CancellationToken token = default)
    {
        logger.LogInformation($"Retrieved message that training of model has been finished");
        return Task.CompletedTask;
    }
}
