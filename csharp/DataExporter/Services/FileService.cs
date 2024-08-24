using Azure.Storage.Blobs;
using Azure.Storage;
using Azure;
using Common.Helpers;
using Common.Exceptions;
using DataExporter.Services.Factory;
using Microsoft.Extensions.Logging;
using Azure.Storage.Blobs.Models;

namespace app.Services;

public interface IFileService
{
    Task UploadToAzure(string containerName, string generatedFileName, CancellationToken token = default);
}

public class FileService : IFileService
{
    private readonly BlobServiceClient _blobServiceClient;
    private readonly ILogger<FileService> _logger;

    public FileService(IConfiguration configuration, IBlobServiceClientFactory blobServiceClientFactory, ILogger<FileService> logger)
    {
        _logger = logger;
        var _accountName = configuration.GetValue<string>("accountName");
        if (string.IsNullOrEmpty(_accountName))
        {
            throw new AccountNameNullException("FileStorage:accountName cannot be NULL");
        }
        var _accountKey = configuration.GetValue<string>("accountKey");
        if (string.IsNullOrEmpty(_accountKey)) {
            throw new AccountKeyNullException("FileStorage:accountKey cannot be NULL");
        }

        _blobServiceClient = blobServiceClientFactory.Create(_accountName, _accountKey);
    }



    private async Task<BlobContainerClient?> EnsureContainer(string containerName, CancellationToken token)
    {
        _logger.LogInformation("[EnsureContainer] get blobContainerClient");
        var containerClient = _blobServiceClient.GetBlobContainerClient(containerName);

        try
        {
            // Create the container
            _logger.LogInformation("[EnsureContainer] Start CreateIfNotExistsAsync");
            await containerClient.CreateIfNotExistsAsync(PublicAccessType.None, null, null, token);
            _logger.LogInformation("[EnsureContainer] Check if existsasync exists");
            var containerExists = await containerClient.ExistsAsync(token);
            if (containerExists.Value)
            {
                _logger.LogInformation("[EnsureContainer] Created container {Name}", containerClient.Name);
                return containerClient;
            }
        }
        catch (RequestFailedException e)
        {
            _logger.LogError(e,"[EnsureContainer] HTTP error code {Status}: {ErrorCode}, {Message}", e.Status, e.ErrorCode, e.Message);
            
        }
        _logger.LogInformation("[EnsureContainer] Pretty weird I end up here ");
        return containerClient;
    }
    private async Task UploadFromFileAsync(
                                BlobContainerClient containerClient,
                                string localFilePath,
                                CancellationToken token)
    {

        string fileName = Path.GetFileName(localFilePath);
        _logger.LogInformation("[UploadFromFileASync] Uploading file {fileName}", fileName);
        BlobClient blobClient = containerClient.GetBlobClient(fileName);
        var uploadResult = await blobClient.UploadAsync(localFilePath, true, token);
        _logger.LogInformation("[UploadFromFileASync] Finished uploading ... result : {uploadResult}", uploadResult);
    }

    public async Task UploadToAzure(string containerName, string generatedFileName, CancellationToken token = default)
    {
        _logger.LogInformation("[UploadToAzure] Start uploading to azure using {containerName} and file {generatedFileName}", containerName, generatedFileName);
        var result = await EnsureContainer(StorageHelpers.ContainerName, token) ?? throw new EnsureContainerException("result is null");
        await UploadFromFileAsync(result, generatedFileName, token);
        _logger.LogInformation("[UploadToAzure] Finished upload to Azure");
    }
}