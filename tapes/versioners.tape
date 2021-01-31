# * "checkout" is run in a sandbox directory
# * All other commands are run inside $dir
versioner disk :: path {
  action checkout > dir {
    if [ ! -e $path ]; then
      echo >&2 "Directory does not exist: $path"
      exit 1
    fi
    ln -s $path'/'* $dir/
  }
  action repo_version > version {
    if [ ! -e $path ]; then
      echo >&2 "Directory does not exist: $path"
      exit 1
    fi
    echo "VERSIONING_UNSUPPORTED" > $version
  }
  action local_version > version date {
    if [ ! -e $path ]; then
      echo >&2 "Directory does not exist: $path"
      exit 1
    fi
    echo "VERSIONING_UNSUPPORTED" > $version
    echo "VERSIONING_UNSUPPORTED" > $date
  }
}

versioner git :: repo ref {
  action checkout > dir {
    git clone $repo $dir
    cd $dir
    git checkout $ref
  }
  action repo_version > version {
    git ls-remote $repo $ref | cut -f1 > $version
  }
  # Used to confirm version after checkout
  action local_version > version date {
    git rev-parse HEAD > $version
    git log -1 | awk '/^Date/{$1=""; print}' > $date
  }
}

versioner pip :: package tag {
  action checkout > dir {
    pip install $package==$tag
  }

  action repo_version > version {
    pip show $package | grep ^Version: | cut -d' ' -f2 > $version ||
      echo "0" > $version
  }

  # Used to confirm version after checkout
  action local_version > version date {
    sacrebleu -V > $version
    echo > $date
  }
}
