# GitToolFetcher

GitToolFetcher is a tool to help managing multiple concurent versions of GitHub hosted projects.

## How to install

To install GitToolFetcher, we recommend pulling directly from PyPi :

```bash
pip install gittoolfetcher
```

This will install GitToolFetcher as well as its necessary dependencies.

### Building from source

If you wish to build & install GitToolFetcher from source instead, use Hatch's usual build command :

```bash
hatch build
```

The built archive will be placed in the "dist" directory as a .whl file.
To install GitToolFetcher, simply install the .whl file using pip.

```bash
pip install dist/gittoolfetcher-*.whl
```

## Command Line Usage

Once installed, a new utility `gittoolfetcher` will be available.

```
usage: gittoolfetcher [-h] [-r] [-s] [-i [INSTALL ...]] [-u [UNINSTALL ...]] [-v VERSION] [-l] [-e COMMAND] REPO_NAME

positional arguments:
  REPO_NAME             Name of the GitHub repo to manage.

options:
  -h, --help                                       show this help message and exit
  -r, --refresh                                    Refresh the available version cache.
  -s, --show                                       Show available project versions.
  -i [INSTALL ...], --install [INSTALL ...]        Install one or more project versions.
  -u [UNINSTALL ...], --uninstall [UNINSTALL ...]  Uninstall one or more project versions.
  -v [VERSION ...], --version [VERSION ...]        Select a version of the project.
  -l, --list                                       List installed project versions.
  -e COMMAND, --execute COMMAND                    Execute a command with the selected version of the project.
```

Here is a typical workflow using GitToolFetcher :

```bash
gittoolfetcher golang/go -r -s -i go1.14,go1.6,go1.23.4 -l
```