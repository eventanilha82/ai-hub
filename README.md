# AI Hub

Aplicacao de chat em `Streamlit` integrada ao `OpenAI Agents SDK`, usando um endpoint compativel com OpenAI hospedado na OCI. O agente preserva contexto com `conversations` remotas e pode usar Tavily via MCP nativo.

## O que o codigo faz

- `app.py` cria a interface estilo chat, permite informar o nome do usuario no sidebar, habilitar Tavily MCP quando necessario, mantem o historico visual no estado do Streamlit e exibe erros no fluxo da conversa.
- `agent.py` monta o cliente OpenAI compativel com OCI usando `OCI_PROJECT`, configura um agente simples, cria uma `conversation` remota por usuario ativo e entrega respostas curtas em streaming para a UI com `conversation_id`.
- Quando habilitado no sidebar, `agent.py` conecta ao servidor MCP do Tavily via `TAVILY_MCP_URL` e passa `mcp_servers=[server]` diretamente ao agente.
- Se `TAVILY_SEARCH_URL` estiver preenchida, o agente prioriza essa fonte nas pesquisas do Tavily.
- `.env` concentra a configuracao do endpoint e da autenticacao.

## Pre-requisitos

- Python `3.11+`
- `uv` instalado
- Acesso a um endpoint compativel com OpenAI na OCI
- Um modelo liberado nesse endpoint
- `OCI_API_KEY`
- `OCI_BASE_URL`
- `OCI_MODEL_ID`
- `OCI_PROJECT`
- `TAVILY_MCP_URL`

## Configuracao

1. Crie o arquivo de ambiente a partir do exemplo:

```bash
cp .env.example .env
```

2. Preencha a configuracao:

```env
OCI_API_KEY=seu-token-aqui
OCI_BASE_URL=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1
OCI_MODEL_ID=xai.grok-4-fast-reasoning
OCI_PROJECT=ocid1.generativeaiproject...
TAVILY_MCP_URL=https://mcp.tavily.com/mcp/?tavilyApiKey=...
TAVILY_SEARCH_URL=https://site-ou-blog-para-priorizar.example/
```

Para obter o `TAVILY_MCP_URL`, acesse [Tavily](https://www.tavily.com/), crie ou entre na sua conta e gere uma API key no dashboard. A documentacao oficial do [Tavily MCP Server](https://docs.tavily.com/documentation/mcp) usa este formato para o servidor MCP remoto:

```env
TAVILY_MCP_URL=https://mcp.tavily.com/mcp/?tavilyApiKey=<sua-api-key>
```

Substitua `<sua-api-key>` pela chave Tavily, normalmente com prefixo `tvly-`. O `TAVILY_SEARCH_URL` e opcional; use-o apenas quando quiser priorizar uma fonte especifica nas buscas.

## Como rodar

1. Instale as dependencias:

```bash
uv sync
```

2. Inicie a aplicacao:

```bash
uv run streamlit run app.py
```

3. Abra o endereco exibido pelo Streamlit no navegador.

## Fluxo da aplicacao

1. O usuario informa o nome no sidebar e clica em `Usar usuario`.
2. A UI transforma esse nome em `memory_subject_id` e cria uma `conversation` remota no runtime principal.
3. Opcionalmente, o usuario habilita `Tavily MCP` no sidebar para permitir pesquisa externa.
4. O `Streamlit` recebe a mensagem do usuario.
5. O app envia apenas a nova mensagem e o `Runner` recebe o `conversation_id` remoto da conversa.
6. Se Tavily estiver habilitado, o app cria `MCPServerStreamableHttp`, monta o agente com `mcp_servers=[server]` e chama o `Runner` com `conversation_id`.
7. A resposta do modelo chega em streaming e e renderizada incrementalmente no chat.
8. Enquanto a sessao do Streamlit estiver ativa e o usuario nao mudar, a UI reaproveita a mesma `conversation` remota.

## Memoria da conversa

- A UI cria uma `conversation` remota com `client.conversations.create(...)` no runtime principal.
- A conversation recebe `metadata={"memory_subject_id": ...}` com o id derivado do nome informado no sidebar.
- Cada resposta usa `Runner.run_streamed(..., conversation_id=...)`.
- O historico visual fica apenas no `st.session_state`; nao ha `SQLiteSession`, banco local ou pasta `data`.
- Limpar a conversa cria outra `conversation` para o usuario atual.
- Trocar o usuario cria outra `conversation` e limpa o historico visual.

## Variaveis de ambiente

| Variavel | Obrigatoria | Descricao |
| --- | --- | --- |
| `OCI_API_KEY` | Sim | Token usado como `Bearer` |
| `OCI_BASE_URL` | Sim | Base URL compativel com a API OpenAI |
| `OCI_MODEL_ID` | Sim | ID do modelo usado pelo agente |
| `OCI_PROJECT` | Sim | Projeto OCI passado como `project` no cliente OpenAI |
| `TAVILY_MCP_URL` | Sim para usar Tavily | URL Streamable HTTP do MCP Tavily, incluindo a API key |
| `TAVILY_SEARCH_URL` | Nao | Fonte prioritaria para buscas via Tavily |
| `AGENT_LOG_LEVEL` | Nao | Nivel de log. Padrao: `INFO` |

## Problemas comuns

### `Variaveis ausentes`

Confira se o `.env` esta no diretorio raiz e se as variaveis obrigatorias foram preenchidas.

### `Falha de autorizacao (401/403)`

- confirme se a `OCI_API_KEY` esta valida
- valide se o `OCI_BASE_URL` aponta para o endpoint compativel com OpenAI que suporta `conversations`

### Nenhuma resposta aparece

- confirme se o modelo em `OCI_MODEL_ID` existe nesse endpoint
- valide se a conta tem permissao para inferencia nesse modelo
- revise logs do terminal onde o Streamlit foi iniciado

### Muito lento ou com pausas longas

- o runtime usa os defaults nativos do SDK e do cliente OpenAI compativel
- o `conversation_id` remoto preserva o contexto da conversa
- o `OCI_MODEL_ID` e usado pelo agente simples
- o chat foi calibrado para responder de forma curta e objetiva
- acompanhe os logs `[CONVERSATION] ...` e `[CHAT] ...`

## Desenvolvimento

Checagens uteis:

```bash
uv run python -m py_compile agent.py app.py
uv run python -m unittest discover -s tests -v
```

Smoke tests reais contra a API:

```bash
RUN_LIVE_AGENT_TESTS=1 uv run python -m unittest tests.test_agent_live -v
```

Cobertura local principal:

```bash
uv run python -m unittest tests.test_agent_unit tests.test_app_unit -v
```

O que os testes cobrem:

- configuracao obrigatoria e reutilizacao do runtime
- criacao de conversation com `memory_subject_id`
- toggle Tavily MCP e `mcp_servers=[server]` quando habilitado
- streaming incremental e fallback por `final_output`
- erros de autenticacao, rate limit, conexao e status HTTP do runtime
- troca de usuario, limpeza do chat, toast e ordenacao do historico
- smoke real do streaming e de dois turnos na mesma conversation

## Observacoes de seguranca

- O arquivo `.env` foi adicionado ao `.gitignore` para evitar commit acidental de segredo.
- Se alguma chave real ja foi versionada ou compartilhada, a acao correta e rotacionar essa credencial.
