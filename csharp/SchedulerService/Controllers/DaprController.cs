using Common.Helpers;
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
    private readonly DaprClient _daprClient;

    public DaprController(ILogger<DaprController> logger, DaprClient daprClient)
    {
        this.logger = logger;
        _daprClient = daprClient;
    }

    [Topic(NameConsts.INFLUX_PUBSUB_NAME, "testreply")]
    [HttpPost("testreply")]
    public IActionResult TestReply([FromBody] RetrieveDataResponse o)
    {
        logger.LogInformation($"TestReply got triggered");
        if (o is not null)
        {
            logger.LogInformation($"Received RetrieveDataResponse response : {o.Success} for id {o.GeneratedFileName} -> {o.StartAiProcess}");
            return Ok();
        }
        return BadRequest();
    }


    [Topic(pubsubName: NameConsts.INFLUX_PUBSUB_NAME, name: NameConsts.INFLUX_FINISHED_RETRIEVE_DATA)]
    [HttpPost("DownloadDataHasFinished")]
    public async Task DownloadDataHasFinished([FromBody] RetrieveDataResponse response)
    {
        logger.LogInformation($"Response value : {JsonConvert.SerializeObject(response)}");
        logger.LogInformation($"Retrieved info that download has been completed {response?.Success} in filename {response?.GeneratedFileName}");
        logger.LogInformation($"Start AI Processing ? {response?.StartAiProcess}");
        var mustTrainModel = false;
        if (response != null && !response.Success)
        {
            logger.LogWarning("Download of data pre training AI model failed !!");
            return;
        }
        if (response != null && response?.StartAiProcess != null && response.StartAiProcess == true)
        {
            logger.LogInformation("Start training model was true => set boolean val");
            mustTrainModel = true;
        }

        if (mustTrainModel)
        {
            logger.LogInformation("Start triggering of downloading data to python training container");
            await _daprClient.PublishEventAsync(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_START_DOWNLOAD_DATA);
            logger.LogInformation("Sent message to AI container to start Downloading data and prepare it to start training model");
        }
        else
        {
            logger.LogInformation("Training model was not needed so skipped");
        }
        logger.LogInformation($"Finished exchanging messages");
    }

    [Topic(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_FINISHED_DOWNLOAD_DATA)]
    [HttpPost("AiDownloadFinishedStartTraining")]
    public async Task AiDownloadFinishedStartTraining()
    {
        logger.LogInformation("Retrieved from python module that download of data has been finished");
        logger.LogInformation("Send message to start training model on AI container");
        await _daprClient.PublishEventAsync(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_START_TRAIN_MODEL);
        logger.LogInformation("Finished sending message to AI container to start training model");

    }

    [Topic(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_FINISHED_TRAIN_MODEL)]
    [HttpPost("AiFinishedTrainingModel")]
    public Task AiFinishedTrainingModel([FromBody] TrainAiModelResponse response)
    {
        logger.LogInformation($"Retrieved message that training of model has been finished");
        logger.LogInformation($"Training was a success: {response.Success}");
        logger.LogInformation("Process finished");
        return Task.CompletedTask;
    }
}
