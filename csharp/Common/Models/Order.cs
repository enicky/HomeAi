namespace Common.Models;
public class Order
{
    public int Id { get; set; }
    public string Title { get; set; } = "";
}

public class OrderReceived
{
    public bool Success { get; set; }
    public int OrderId { get; set; }
}
