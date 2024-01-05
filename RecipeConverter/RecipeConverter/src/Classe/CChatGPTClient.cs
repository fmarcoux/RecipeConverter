using RecipeConverter.src.Interface;  
using RecipeConverter.src.DTO.ChatGptRequest;
using Google.Apis.Http;
using System.Net.Http.Json;
using Microsoft.Extensions.Configuration;

namespace RecipeConverter.src.Classe
{
    public class CChatGPTClient : IChatGptClient
    {
        private readonly HttpClient m_httpClient;
        private readonly string m_apiKey;
        
        public CChatGPTClient(IConfiguration config)
        {
            if (String.IsNullOrEmpty(config["ChatGPTApiURL"])) throw new ArgumentNullException("ChatGPTApiURL is null or empty");
            if (String.IsNullOrEmpty(config["ChatGPTApiKey"])) throw new ArgumentNullException("ChatGPTApiKey is null or empty");


            m_httpClient = new HttpClient();
            m_httpClient.BaseAddress = new Uri(config["ChatGPTApiURL"]!);
            m_apiKey = config["ChatGPTApiKey"]!;
            m_httpClient.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer",m_apiKey);
        }

        public async Task<string> QueryChaGPT(ChatGPTRequestDTO text)
        {
            HttpContent content = JsonContent.Create(text) ;
            var response = await m_httpClient.PostAsync(String.Empty,content);
            Console.WriteLine(response.StatusCode);
            return response.Content.ReadAsStringAsync().Result;
        }
    }
}
