using RecipeConverter.src.Classe;
using Microsoft.Extensions.Configuration;
using RecipeConverter.src.DTO.ChatGptRequest;

namespace RecipeConverterTest
{
    [TestClass]
    public class CChatGPTClientTest
    {
        private static readonly string m_projectPath = DirectoryUtils.GetProjectRootDiretory();
        private static readonly string m_appsettingsPath = Path.Combine(Directory.GetParent(m_projectPath)!.FullName,"RecipeConverter","appsettings.json");


        [TestMethod]
        public void CChatGPTClientTest_RandomSingleRequestReturnNotEmptyBody()
        {
            CChatGPTClient client = CChatGPTClientConstructor();
            ChatGPTRequestDTO request = new ChatGPTRequestDTO
            {
                Model = "gpt-3.5-turbo",
                Messages = new List<Message> { 
                    new Message { 
                                    Role= "user",
                                    Content = "Quel est la couleur du cheval blanc de napoleon"
                                }
                }
            };  
            string response = client.QueryChaGPT(request).Result;
            Assert.IsNotNull(response); 
            Assert.IsTrue(response.Length > 0);
            Console.WriteLine(response);
            
        }
        
        private CChatGPTClient CChatGPTClientConstructor()
        {
            IConfiguration config = new ConfigurationBuilder()
                .AddJsonFile(m_appsettingsPath)
                .Build();
            CChatGPTClient client = new CChatGPTClient(config);
            Assert.IsNotNull(client);
            return client;
        }
    }
}