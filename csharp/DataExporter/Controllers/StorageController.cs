using Common.Helpers;
using Common.Models.AI;
using Common.Services;
using Dapr;
using Dapr.Client;
using Microsoft.AspNetCore.Mvc;

namespace DataExporter.Controllers;

public class StorageController(ILogger<StorageController> logger, IFileService fileService)
    : ControllerBase
{
    [Topic(pubsubName: NameConsts.AI_PUBSUB_NAME, name: NameConsts.AI_START_UPLOAD_MODEL)]
    [HttpPost(NameConsts.AI_START_UPLOAD_MODEL)]
    public async Task StartUploadingModelToAzure(
        [FromBody] StartUploadModel startUploadModel,
        [FromServices] DaprClient daprClient,
        CancellationToken token
    )
    {
        logger.LogInformation("Trigger received to upload model {ModelPath} to azure", startUploadModel.ModelPath);
        logger.LogInformation("ModelPath : {ModelPath}, TriggerMoment : {DateTime}", startUploadModel.ModelPath, startUploadModel.TriggerMoment);
        try
        {
            string fileName = Path.GetFileName(startUploadModel.ModelPath);
            logger.LogInformation("FileName retrieved from ... {ModelPath} {FileName}", startUploadModel.ModelPath, fileName);
            var generatedFileName = $"{DateTime.Now.ToString("yyyyMMdd")}-{fileName}";
            var targetFolder = Path.GetDirectoryName(startUploadModel.ModelPath);
            var fullPath = Path.Join(targetFolder, generatedFileName);
            logger.LogInformation("Renaming file to {FullPath}", fullPath);
            if (System.IO.File.Exists(fullPath))
            {
                logger.LogInformation("File {FullPath} already exists. We Can override the target file", fullPath);
            }
            System.IO.File.Copy(startUploadModel.ModelPath, fullPath, true);
            logger.LogInformation("Start uploading file");
            await fileService.UploadToAzure(StorageHelpers.ModelContainerName,fullPath,token);
            logger.LogInformation("Finished uploading file to Azure");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "There was an error processing the model to azure");
        }
    }
}
