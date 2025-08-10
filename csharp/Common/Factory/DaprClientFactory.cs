
using Dapr.Client;

namespace Common.Factory;

public interface IDaprClientWrapper
{
    Task PublishEventAsync(string pubsubName, string topicName, object data, CancellationToken cancellationToken = default);
    Task<T> InvokeMethodAsync<T>(HttpMethod method, string appId, string methodName, CancellationToken cancellationToken = default);
}

public class DaprClientWrapper : IDaprClientWrapper
{
    private readonly DaprClient _client;
    public DaprClientWrapper(DaprClient client)
    {
        _client = client;
    }
    public Task PublishEventAsync(string pubsubName, string topicName, object data, CancellationToken cancellationToken = default)
        => _client.PublishEventAsync(pubsubName, topicName, data, cancellationToken);
    public Task<T> InvokeMethodAsync<T>(HttpMethod method, string appId, string methodName, CancellationToken cancellationToken = default)
        => _client.InvokeMethodAsync<T>(method, appId, methodName, cancellationToken);
}

public interface IDaprClientFactory
{
    IDaprClientWrapper CreateClient();
}

public class DaprClientFactory : IDaprClientFactory
{
    public IDaprClientWrapper CreateClient() => new DaprClientWrapper(new DaprClientBuilder().Build());
}