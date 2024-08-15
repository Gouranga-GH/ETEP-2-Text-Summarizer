import setuptools  # Importing the setuptools module to assist in packaging the Python project.

# Opening and reading the content of the README.md file to use as the long description.
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()  # Reading the file content to be used later.

# Setting the version of the package.
__version__ = "0.0.0"

# Defining variables for repository and author information.
REPO_NAME = "ETEP-Modular-Coding-Docker-AWS"  # The name of the repository on GitHub.
AUTHOR_USER_NAME = "Gouranga-GH"  # GitHub username of the author.
SRC_REPO = "textSummarizer"  # The name of the main source folder.
AUTHOR_EMAIL = "post.gourang@gmail.com"  # The email address of the author.

# Calling setuptools.setup() to define the package details.
setuptools.setup(
    name=SRC_REPO,  # The name of the package.
    version=__version__,  # The version of the package.
    author=AUTHOR_USER_NAME,  # The author's name.
    author_email=AUTHOR_EMAIL,  # The author's email address.
    description="A python package for NLP app",  # A short description of the package.
    long_description=long_description,  # The long description read from README.md.
    long_description_content="text/markdown",  # Specifies that the long description is in Markdown format.
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # The URL to the GitHub repository.
    project_urls={  # Additional URLs related to the project, such as the issue tracker.
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},  # Specifies that packages are located inside the "src" directory.
    packages=setuptools.find_packages(where="src")  # Automatically finds all packages inside the "src" directory.
)
