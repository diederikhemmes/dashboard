import sys
import streamlit
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "dashboard.py"]
    sys.exit(stcli.main())


