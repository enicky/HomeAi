using System.Text.Json;
using System.Xml.Linq;

if (args.Length != 2)
{
    Console.Error.WriteLine("Usage: CoverletJsonToOpenCover <input.coverage.json> <output.opencover.xml>");
    Environment.Exit(1);
}

string inputPath = args[0];
string outputPath = args[1];

if (!File.Exists(inputPath))
{
    Console.Error.WriteLine($"Error: File '{inputPath}' not found.");
    Environment.Exit(1);
}

var json = File.ReadAllText(inputPath);
var doc = JsonDocument.Parse(json);

XNamespace ns = "http://schemas.microsoft.com/visualstudio/coverage/opensource";
var coverage = new XElement(ns + "CoverageSession",
    new XElement(ns + "Modules")
);

foreach (var assembly in doc.RootElement.EnumerateObject())
{
    var moduleEl = new XElement(ns + "Module",
        new XElement(ns + "ModulePath", assembly.Name),
        new XElement(ns + "ModuleName", assembly.Name),
        new XElement(ns + "Files"),
        new XElement(ns + "Classes")
    );

    var fileIdMap = new Dictionary<string, int>();
    int fileCounter = 1;

    foreach (var classProp in assembly.Value.EnumerateObject())
    {
        var classEl = new XElement(ns + "Class",
            new XElement(ns + "FullName", classProp.Name),
            new XElement(ns + "Methods")
        );

        foreach (var methodProp in classProp.Value.EnumerateObject())
        {
            var methodData = methodProp.Value;
            var methodName = methodProp.Name;

            var methodEl = new XElement(ns + "Method",
                new XAttribute("name", methodName),
                new XAttribute("class", classProp.Name),
                new XElement(ns + "SequencePoints")
            );

            if (methodData.TryGetProperty("Lines", out var linesElement))
            {
                foreach (var lineProp in linesElement.EnumerateObject())
                {
                    int lineNumber = int.Parse(lineProp.Name);
                    int visits = lineProp.Value.GetInt32();

                    string fakeFilePath = assembly.Name + ".cs"; // Placeholder since JSON may not store file info
                    if (!fileIdMap.ContainsKey(fakeFilePath))
                        fileIdMap[fakeFilePath] = fileCounter++;

                    methodEl.Element(ns + "SequencePoints")!.Add(
                        new XElement(ns + "SequencePoint",
                            new XAttribute("vc", visits),
                            new XAttribute("sl", lineNumber),
                            new XAttribute("el", lineNumber),
                            new XAttribute("fileid", fileIdMap[fakeFilePath]),
                            new XAttribute("nseq", 0),
                            new XAttribute("bev", 0),
                            new XAttribute("sc", 0),
                            new XAttribute("ec", 0)
                        )
                    );
                }
            }

            classEl.Element(ns + "Methods")!.Add(methodEl);
        }

        foreach (var file in fileIdMap)
        {
            moduleEl.Element(ns + "Files")!.Add(
                new XElement(ns + "File",
                    new XAttribute("uid", file.Value),
                    new XAttribute("fullPath", file.Key)
                )
            );
        }

        moduleEl.Element(ns + "Classes")!.Add(classEl);
    }

    coverage.Element(ns + "Modules")!.Add(moduleEl);
}

var xdoc = new XDocument(
    new XDeclaration("1.0", "utf-8", "yes"),
    coverage
);

Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
xdoc.Save(outputPath);

Console.WriteLine($"OpenCover report generated: {outputPath}");
