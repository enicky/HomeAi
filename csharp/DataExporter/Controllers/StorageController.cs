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
        logger.LogInformation($"[StorageController:StartUploadingModelToAzure] Start uploading model {startUploadModel.ModelPath} to azure");
        try
        {
            string[] allfiles = Directory.GetFiles("/app/checkpoints/models", "*.*", SearchOption.AllDirectories);
            foreach (var item in allfiles )
            {
                logger.LogInformation($" --> {item} ...");
            }

            string fileName = Path.GetFileName(startUploadModel.ModelPath);
            logger.LogInformation($"FileName retrieved from ... {startUploadModel.ModelPath} {fileName}");
            var generatedFileName = $"{DateTime.Now.ToString("yyyyMMdd")}-{fileName}";
            logger.LogInformation(
                $"[StorageController:StartUploadingModelToAzure] Renaming file to {generatedFileName}"
            );
            if (System.IO.File.Exists(generatedFileName))
            {
                logger.LogInformation($"File {generatedFileName} already exists. We Can override the target file");
                //System.IO.File.Delete(generatedFileName);
                //logger.LogInformation("File deleted");
            }
            System.IO.File.Copy(startUploadModel.ModelPath, generatedFileName, true);
            logger.LogInformation(
                "[StorageController:StartUploadingModelToAzure] Start uploading file"
            );
            await fileService.UploadToAzure(
                StorageHelpers.ModelContainerName,
                generatedFileName,
                token
            );
            logger.LogInformation(
                "[StorageController:StartUploadingModelToAzure] Finished uploading file to Azure"
            );
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "There was an error processing the model to azure");
        }
        // first Rename file to current datetime = model
    }
}
