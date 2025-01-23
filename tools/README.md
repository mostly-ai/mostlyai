# Tools

## model extension

While `datamodel-codegen` is a very handy tool for converting OpenAPI definitions into
pydantic objects (see `mostlyai/sdk/client/domain.py`), it lacks the direct ability of adding extra functionality (e.g. methods)
to the classes it creates. For example, having `Generator.add_table(...)` is out of its scope, but
there's a trick being used here:
- `datamodel-codegen` works with `Jinja2` templates, and custom templates can be specified
- That specific template is located in `custom_template/pydantic_v2/BaseModel.jinja2`
- That template is created based on `tools/model.py`, which contains the functionality to add to existing classes
- Running `extend_model.py` updates the template based on `tools/model.py`

tl;dr the content of `tools/model.py` is being stitched to what `datamodel-codegen` generates natively.
Moreover, all of that happens by simply running `make gen-public-model`

## Updating public model (based on `public-api.yaml`)

`make gen-public-model` does all the required actions to rewrite and format `mostlyai/sdk/client/domain.py`. The relevant code sections from `tools/model.py` are stitched into the resulted `mostlyai/sdk/client/domain.py`
