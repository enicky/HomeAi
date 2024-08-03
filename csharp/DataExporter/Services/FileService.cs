using Azure.Storage.Blobs;
using Azure.Storage;
using Azure;
using Common.Helpers;
using Common.Exceptions;

namespace app.Services;

public interface IFileService
{
    Task UploadToAzure(string containerName, string generatedFileName);
}

public class FileService : IFileService
{
    private readonly BlobServiceClient _blobServiceClient;
    public FileService(IConfiguration configuration)
    {
        var _accountName = configuration.GetValue<string>("accountName");
        if (string.IsNullOrEmpty(_accountName))
        {
            throw new NullReferenceException("FileStorage:accountName cannot be NULL");
        }
        var _accountKey = configuration.GetValue<string>("accountKey");
        if (string.IsNullOrEmpty(_accountKey)) throw new NullReferenceException("FileStorage:accountKey cannot be NULL");

        _blobServiceClient = GetBlobServiceClient(_accountName, _accountKey);
    }

    private BlobServiceClient GetBlobServiceClient(string accountName, string accountKey)
    {
        StorageSharedKeyCredential sharedKeyCredential = new StorageSharedKeyCredential(accountName, accountKey);
        string blobUri = "https://" + accountName + ".blob.core.windows.net";

        var client = new BlobServiceClient(new Uri(blobUri), sharedKeyCredential);
        return client;
    }

    private async Task<BlobContainerClient?> EnsureContainer(string containerName)
    {
        var containerClient = _blobServiceClient.GetBlobContainerClient(containerName);

        try
        {
            // Create the container

            await containerClient.CreateIfNotExistsAsync();
            if (await containerClient.ExistsAsync())
            {
                Console.WriteLine("Created container {0}", containerClient.Name);
                return containerClient;
            }
        }
        catch (RequestFailedException e)
        {
            Console.WriteLine("HTTP error code {0}: {1}", e.Status, e.ErrorCode);
            Console.WriteLine(e.Message);
        }

        return containerClient;
    }
    private static async Task UploadFromFileAsync(
                                BlobContainerClient containerClient,
                                string localFilePath)
    {

        string fileName = Path.GetFileName(localFilePath);
        BlobClient blobClient = containerClient.GetBlobClient(fileName);

        await blobClient.UploadAsync(localFilePath, true);
    }

    public async Task UploadToAzure(string containerName, string generatedFileName)
    {
        var result = await EnsureContainer(StorageHelpers.ContainerName) ?? throw new EnsureContainerException("result is null");
        await UploadFromFileAsync(result, generatedFileName);
    }
}