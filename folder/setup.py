from setuptools import setup, find_packages

setup(
    name="chatbot-flask",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask',
        'flask-sqlalchemy',
        'psycopg2-binary',
    ],
)