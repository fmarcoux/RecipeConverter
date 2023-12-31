using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReceipeConverter.src.Interface
{
    public interface ITandorApiClient
    {
        public HttpResponseMessage CreateRecipe();
    }
}
