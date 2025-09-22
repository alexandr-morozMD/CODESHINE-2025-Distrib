# AI Orchestrator

Приложение на Streamlit, которое оркестрирует агентов для подготовки маркетинговой кампании и сохраняет артефакты в `artifacts/`.

## Требования
- Python 3.10+
- [Ollama](https://ollama.com/) (локальная установка)
- Свободные ~3 ГБ под модель `phi3:mini` или другую поддержку Ollama

## Подготовка окружения (один раз)
```bash
./scripts/setup.sh
```
Скрипт создаст `.venv`, обновит `pip` и установит зависимости из `requirements.txt`.

## Запуск приложения
```bash
./scripts/run.sh
```
Скрипт:
- поднимает `ollama serve`, если он ещё не запущен (логи пишутся в `ollama.log`),
- активирует `.venv`,
- запускает Streamlit с моделью Ollama `phi3:mini` по умолчанию,
- подавляет интерактивный запрос Streamlit на сбор статистики.

После запуска откройте ссылку из консоли (обычно `http://localhost:8501`). Остановить приложение можно `Ctrl+C`; если `ollama serve` был запущен скриптом, он завершится автоматом.

## Streamlit Cloud
- В разделе Secrets скопируйте значения из `.streamlit/secrets.example.toml` (LLM_BACKEND=`"gemini"`, GEMINI_API_KEY=`"AIzaSy..."`, GEMINI_MODEL=`"gemini-1.5-flash"`).
- Убедитесь, что в `requirements.txt` присутствуют `google-generativeai`, `streamlit`, `langgraph`, `jinja2` (уже включены).
- После деплоя `agents/_llm.py` автоматически переключится на Gemini, локально оставьте `.env` с `LLM_BACKEND=ollama`.

### Кастомизация
- Используйте другую модель: `OLLAMA_MODEL=llama3.1:8b ./scripts/run.sh`.
- Дополнительные документы для RAG можно указать в UI при запуске кампании.

## Полезные файлы и журналы
- `streamlit.log` — лог интерфейса (создаётся самим Streamlit).
- `ollama.log` — лог Ollama, когда запускается через `scripts/run.sh`.
- `artifacts/<run-id>` — сгенерированные артефакты кампаний (игнорируются Git).

## Типичные проблемы
- **Долгая генерация** — проверьте, что модель загружена (`ollama list`) и хватает ресурсов.
- **Первый отклик Ollama идёт очень долго** — увеличьте таймаут через `OLLAMA_TIMEOUT=180 ./scripts/run.sh`.
- **Streamlit не открывается** — убедитесь, что порт 8501 свободен, или задайте `--server.port` через переменную `STREAMLIT_SERVER_PORT` перед запуском.
- **Не найден Ollama** — добавьте путь к Ollama в `PATH` или установите его согласно документации Ollama.

## Остановка приложения
ps aux | grep streamlit
kill -9 <PID>

## Пуск 
streamlit run yourscript.py