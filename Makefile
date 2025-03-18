# Internal Variables
PUBLIC_API_FULL_URL = https://raw.githubusercontent.com/mostly-ai/mostly-openapi/refs/heads/main/public-api.yaml
PUBLIC_API_OUTPUT_PATH = mostlyai/sdk/domain.py

# Targets
.PHONY: help
help: ## show definition of each function
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: gen-public-model
gen-public-model: ## build pydantic models for public api
	@echo "Updating custom Jinja2 templates"
	python tools/extend_model.py
	@echo "Generating Pydantic models from $(PUBLIC_API_FULL_URL)"
	#datamodel-codegen --url $(PUBLIC_API_FULL_URL) $(COMMON_OPTIONS)
	datamodel-codegen --input ../mostly-app-v2/public-api/public-api.yaml $(COMMON_OPTIONS)
	python tools/postproc_domain.py
	uv run --no-sync pre-commit run --all-files

# Common options for both targets
COMMON_OPTIONS = \
	--input-file-type openapi \
	--output $(PUBLIC_API_OUTPUT_PATH) \
	--snake-case-field \
	--target-python-version 3.10 \
	--use-schema-description \
	--use-union-operator \
	--use-standard-collections \
	--field-constraints \
	--collapse-root-models \
	--use-one-literal-as-default \
	--enum-field-as-literal one \
	--use-subclass-enum \
	--output-model-type pydantic_v2.BaseModel \
	--base-class mostlyai.sdk.client.base.CustomBaseModel \
	--custom-template-dir tools/custom_template

.PHONY: clean
clean: ## Remove .gitignore files
	git clean -fdX

# Variables for docker-run
HOST_PORT ?= 8080
HOST_PATH ?=

.PHONY: docker-build
docker-build: ## Build the docker image
	DOCKER_BUILDKIT=1 docker build . --platform=linux/amd64 -t mostlyai/mostlyai

.PHONY: docker-run
docker-run: ## Start the docker container
	@echo "Mapping port: $(HOST_PORT) (host) <-> 8080 (container)"
	@# here we have to make sure .venv folder is set as an anonymous volume, so that it will not be overwritten by a bind mount
	@# ref: https://docs.astral.sh/uv/guides/integration/docker/#mounting-the-project-with-docker-run
	@if [ -z "$(HOST_PATH)" ]; then \
            docker run --platform=linux/amd64 -it -p $(HOST_PORT):8080 \
            -v ~/.cache/huggingface:/opt/app-root/src/.cache/huggingface \
            mostlyai/mostlyai ; \
        else \
            if [ ! -d $(HOST_PATH) ]; then \
                echo "Failed to mount volume: $(HOST_PATH) does not exist"; \
                exit 1; \
            fi; \
            REAL_PATH=$$(realpath $(HOST_PATH)); \
            BASE_NAME=$$(basename $$REAL_PATH); \
            MOUNT_ARGS="--mount type=bind,source=$$REAL_PATH,target=/opt/app-root/src/$$BASE_NAME"; \
            echo "Mounting volume: $$REAL_PATH (host) <-> /opt/app-root/src/$$BASE_NAME (container)"; \
            docker run --platform=linux/amd64 --rm -it -p $(HOST_PORT):8080 \
              -v ~/.cache/huggingface:/opt/app-root/src/.cache/huggingface \
              -v /opt/app-root/src/mostlyai/.venv \
              $$MOUNT_ARGS mostlyai/mostlyai ; \
        fi;

# Default files to update
PYPROJECT_TOML = pyproject.toml
INIT_FILE = mostlyai/sdk/__init__.py

# Internal Variables for Release Workflow
BUMP_TYPE ?= patch
CURRENT_VERSION := $(shell grep -m 1 'version = ' $(PYPROJECT_TOML) | sed -e 's/version = "\(.*\)"/\1/')
# Assuming current_version is already set from pyproject.toml
NEW_VERSION := $(shell echo $(CURRENT_VERSION) | awk -F. -v bump=$(BUMP_TYPE) '{ \
    if (bump == "patch") { \
        printf("%d.%d.%d", $$1, $$2, $$3 + 1); \
    } else if (bump == "minor") { \
        printf("%d.%d.0", $$1, $$2 + 1); \
    } else if (bump == "major") { \
        printf("%d.0.0", $$1 + 1); \
    } else { \
        print "Error: Invalid BUMP_TYPE. Expected patch, minor or major. Input was BUMP_TYPE=" bump; \
        exit 1; \
    } \
}')


# Targets for Release Workflow/Automation
.PHONY: update-version-gh release-pypi docs

update-version-gh: pull-main bump-version update-vars-version create-branch ## Update version in GitHub: pull main, bump version, create and push the new branch

release-pypi: clean-dist pull-main build confirm-upload upload-pypi docs  ## Release to PyPI: pull main, build and upload to PyPI

pull-main: # Pull main branch
	# stash changes
	@git stash
	# switch to main branch
	@git checkout main
	# fetch latest changes
	@git fetch origin main
	# get a clean copy of main branch
	@git reset --hard origin/main
	# clean
	@git clean -fdX

bump-version: # Bump version (default: patch, options: patch, minor, major)
	@echo "Bumping $(BUMP_TYPE) version from $(CURRENT_VERSION) to $(NEW_VERSION)"
	@echo "Replaces $(CURRENT_VERSION) to $(NEW_VERSION) in $(PYPROJECT_TOML)"
	@echo "Replaces $(CURRENT_VERSION) to $(NEW_VERSION) in $(INIT_FILE)"
	@echo "Current directory: $(shell pwd)"
    # Check if current version was found
	@if [ -z "$(CURRENT_VERSION)" ]; then \
        echo "Error: Could not find current version in $(PYPROJECT_TOML)"; \
        exit 1; \
    fi
    # Replace the version in pyproject.toml
	@if [[ "$(shell uname -s)" == "Darwin" ]]; then \
        sed -i '' 's/version = "$(CURRENT_VERSION)"/version = "$(NEW_VERSION)"/g' $(PYPROJECT_TOML); \
        sed -i '' 's/__version__ = "$(CURRENT_VERSION)"/__version__ = "$(NEW_VERSION)"/g' $(INIT_FILE); \
    else \
        sed -i 's/version = "$(CURRENT_VERSION)"/version = "$(NEW_VERSION)"/g' $(PYPROJECT_TOML); \
        sed -i 's/__version__ = "$(CURRENT_VERSION)"/__version__ = "$(NEW_VERSION)"/g' $(INIT_FILE); \
    fi

update-vars-version: # Update the required variables after bump
	$(eval VERSION := $(shell python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"))
	$(eval BRANCH := verbump_$(shell echo $(VERSION) | tr '.' '_'))
	$(eval TAG := $(VERSION))
	@echo "Updated VERSION to $(VERSION), BRANCH to $(BRANCH), TAG to $(TAG)"

create-branch: # Create verbump_{new_ver} branch
	@git checkout -b $(BRANCH)
	@echo "Created branch $(BRANCH)"
	# commit the version bump
	@git add $(INIT_FILE)
	@git add $(PYPROJECT_TOML)
	@git commit -m "Version Bump to $(VERSION)"
	@echo "Committed version bump to $(VERSION)"
	@git push --set-upstream origin $(BRANCH)
	@echo "Pushed branch $(BRANCH) to origin"

clean-dist: # Remove "volatile" directory dist
	@rm -rf dist
	@echo "Cleaned up dist directory"

build: # Build the project and create the dist directory if it doesn't exist
	@mkdir -p dist
	@uv build
	@echo "Built the project"
	@twine check --strict dist/*
	@echo "Project is checked"

confirm-upload: # Confirm before the irreversible zone
	@echo "Are you sure you want to upload to PyPI? (yes/no)"
	@read ans && [ $${ans:-no} = yes ]

upload-pypi: confirm-upload # Upload to PyPI (ensure the token is present in .pypirc file before running upload)
	@twine upload dist/*$(VERSION)* --verbose
	@echo "Uploaded version $(VERSION) to PyPI"

docs: ## Update docs site
	@mkdocs gh-deploy
	@echo "Deployed docs"
