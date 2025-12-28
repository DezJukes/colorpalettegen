## Setup

1. Create a virtual environment
```bash
python -m venv .venv
```

2. Activate the virtual environment

- Windows (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```
- Windows (CMD)
```cmd
.venv\Scripts\activate
```
- macOS / Linux
```bash
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```


## Running Frontend

1. Install dependencies
```
npm install
```

2. Run the frontend
```
npm run start
```

## Running Backend

1. Go to directory
```
cd backend
```

2. Create virtual environment
```
python -m venv venv
```

3. Activate virtual environment
```
.\venv\Scripts\activate
```

4. Install requirements
```
pip install -r .\..\requirements.txt
```

5. Install fastapi and python-multipart
```
pip install fastapi uvicorn
```

```
pip install python-multipart
```

6. Running backend
```
uvicorn main:app --reload
```