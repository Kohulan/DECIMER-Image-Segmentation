import {FileWithPath} from './file';
import {fromEvent} from './file-selector';


it('returns a Promise', async () => {
    const evt = new Event('test');
    expect(fromEvent(evt)).toBeInstanceOf(Promise);
});

it('should return an empty array if the passed event is not a DragEvent', async () => {
    const evt = new Event('test');
    const files = await fromEvent(evt);
    expect(files).toHaveLength(0);
});

it('should return the evt {target} {files} if the passed event is an input evt', async () => {
    const name = 'test.json';
    const mockFile = createFile(name, {ping: true}, {
        type: 'application/json'
    });
    const evt = inputEvtFromFiles(mockFile);

    const files = await fromEvent(evt);
    expect(files).toHaveLength(1);
    expect(files.every(file => file instanceof File)).toBe(true);

    const [file] = files as FileWithPath[];

    expect(file.name).toBe(mockFile.name);
    expect(file.type).toBe(mockFile.type);
    expect(file.size).toBe(mockFile.size);
    expect(file.lastModified).toBe(mockFile.lastModified);
    expect(file.path).toBe(name);
});

it('should return {files} from DataTransfer if {items} is not defined (e.g. IE11)', async () => {
    const name = 'test.json';
    const mockFile = createFile(name, {ping: true}, {
        type: 'application/json'
    });
    const evt = dragEvt([mockFile]);

    const files = await fromEvent(evt);
    expect(files).toHaveLength(1);
    expect(files.every(file => file instanceof File)).toBe(true);

    const [file] = files as FileWithPath[];

    expect(file.name).toBe(mockFile.name);
    expect(file.type).toBe(mockFile.type);
    expect(file.size).toBe(mockFile.size);
    expect(file.lastModified).toBe(mockFile.lastModified);
    expect(file.path).toBe(name);
});

it('should return an empty array if the evt {target} has no {files} prop', async () => {
    const evt = inputEvtFromFiles();
    const files = await fromEvent(evt);
    expect(files).toHaveLength(0);
});

it('should return files from DataTransfer {items} if the passed event is a DragEvent', async () => {
    const name = 'test.json';
    const mockFile = createFile(name, {ping: true}, {
        type: 'application/json'
    });
    const item = dataTransferItemFromFile(mockFile);
    const evt = dragEvtFromFilesAndItems([], [item]);

    const files = await fromEvent(evt);
    expect(files).toHaveLength(1);
    expect(files.every(file => file instanceof File)).toBe(true);

    const [file] = files as FileWithPath[];

    expect(file.name).toBe(mockFile.name);
    expect(file.type).toBe(mockFile.type);
    expect(file.size).toBe(mockFile.size);
    expect(file.lastModified).toBe(mockFile.lastModified);
    expect(file.path).toBe(name);
});

it('skips DataTransfer {items} that are of kind "string"', async () => {
    const name = 'test.json';
    const mockFile = createFile(name, {ping: true}, {
        type: 'application/json'
    });
    const f = dataTransferItemFromFile(mockFile);
    const str = dataTransferItemFromStr('test');
    const evt = dragEvtFromItems([str, f]);

    const files = await fromEvent(evt);
    expect(files).toHaveLength(1);

    const [file] = files as FileWithPath[];

    expect(file.name).toBe(mockFile.name);
    expect(file.type).toBe(mockFile.type);
    expect(file.size).toBe(mockFile.size);
    expect(file.lastModified).toBe(mockFile.lastModified);
    expect(file.path).toBe(name);
});

it('can read a tree of directories recursively and return a flat list of FileWithPath objects', async () => {
    const mockFiles = sortFiles([
        createFile('ping.json', {ping: true}),
        createFile('pong.json', {pong: true}),
        createFile('foo.json', {foo: true}),
        createFile('bar.json', {bar: true}),
        createFile('john.json', {john: true}),
        createFile('jane.json', {jane: true})
    ]);
    const [f1, f2, f3, f4, f5, f6] = mockFiles;
    const [f7, f8] = [
        createFile('.DS_Store', {macOs: true}),
        createFile('Thumbs.db', {windows: true})
    ];
    const evt = dragEvtFromItems([
        dataTransferItemFromEntry(fileSystemFileEntryFromFile(f1), f1),
        dataTransferItemFromEntry(fileSystemFileEntryFromFile(f2), f2),
        dataTransferItemFromEntry(fileSystemDirEntryFromFile([
            fileSystemFileEntryFromFile(f3),
            fileSystemDirEntryFromFile([
                fileSystemFileEntryFromFile(f4)
            ]),
            fileSystemFileEntryFromFile(f5)
        ], 2)),
        dataTransferItemFromEntry(fileSystemFileEntryFromFile(f6), f6),
        dataTransferItemFromEntry(fileSystemFileEntryFromFile(f7), f7),
        dataTransferItemFromEntry(fileSystemFileEntryFromFile(f8), f8)
    ]);

    const items = await fromEvent(evt);
    const files = sortFiles(items as FileWithPath[]);
    expect(files).toHaveLength(6);
    expect(files.every(file => file instanceof File)).toBe(true);
    expect(files.every(file => typeof file.path === 'string')).toBe(true);
    expect(files).toEqual(mockFiles);
});

it('returns the DataTransfer {items} if the DragEvent {type} is not "drop"', async () => {
    const name = 'test.json';
    const mockFile = createFile(name, {ping: true}, {
        type: 'application/json'
    });
    const item = dataTransferItemFromFile(mockFile);
    const evt = dragEvtFromItems(item, 'dragenter');

    const items = await fromEvent(evt);
    expect(items).toHaveLength(1);

    const [itm] = items as DataTransferItem[];

    expect(itm.kind).toBe(item.kind);
    expect(itm.kind).toBe('file');
});

it(
    'filters DataTransfer {items} if the DragEvent {type} is not "drop" and DataTransferItem {kind} is "string"',
    async () => {
        const name = 'test.json';
        const mockFile = createFile(name, {ping: true}, {
            type: 'application/json'
        });
        const file = dataTransferItemFromFile(mockFile);
        const str = dataTransferItemFromStr('test');
        const evt = dragEvtFromItems([file, str], 'dragenter');

        const items = await fromEvent(evt);
        expect(items).toHaveLength(1);

        const [item] = items as DataTransferItem[];

        expect(item.kind).toBe(file.kind);
        expect(item.kind).toBe('file');
    }
);

it('filters thumbnail cache files', async () => {
    const mockFile = createFile('Thumbs.db', {ping: true}, {
        type: 'text/plain'
    });
    const evt = dragEvt([mockFile]);
    const items = await fromEvent(evt);
    expect(items).toHaveLength(0);
});

it('should throw if reading dir entries fails', async done => {
    const mockFiles = sortFiles([
        createFile('ping.json', {ping: true}),
        createFile('pong.json', {pong: true})
    ]);
    const [f1, f2] = mockFiles;
    const evt = dragEvtFromItems([
        dataTransferItemFromEntry(fileSystemDirEntryFromFile([
            fileSystemFileEntryFromFile(f1),
            fileSystemFileEntryFromFile(f2)
        ], 1, 1))
    ]);

    try {
        await fromEvent(evt);
        done.fail('Getting the files should have failed');
    } catch (err) {
        done();
    }
});

it('should throw if reading file entry fails', async done => {
    const mockFiles = sortFiles([
        createFile('ping.json', {ping: true}),
        createFile('pong.json', {pong: true})
    ]);
    const [f1, f2] = mockFiles;
    const evt = dragEvtFromItems([
        dataTransferItemFromEntry(fileSystemDirEntryFromFile([
            fileSystemFileEntryFromFile(f1),
            fileSystemFileEntryFromFile(f2, 'Oops :(')
        ], 1, 1))
    ]);

    try {
        await fromEvent(evt);
        done.fail('Getting the files should have failed');
    } catch (err) {
        done();
    }
});

it('should throw if DataTransferItem is not a File', async done => {
    const item = dataTransferItem(null, 'file');
    const evt = dragEvtFromFilesAndItems([], [item]);

    try {
        await fromEvent(evt);
        done.fail('Getting the files should have failed');
    } catch (err) {
        done();
    }
});


function dragEvtFromItems(items: DataTransferItem | DataTransferItem[], type: string = 'drop'): DragEvent {
    return {
        type,
        dataTransfer: {
            items: Array.isArray(items) ? items : [items]
        }
    } as any;
}

function dragEvt(files?: File[], items?: DataTransferItem[], type: string = 'drop'): DragEvent {
    return {
        type,
        dataTransfer: {items, files}
    } as any;
}

function dragEvtFromFilesAndItems(files: File[], items: DataTransferItem[], type: string = 'drop'): DragEvent {
    return {
        type,
        dataTransfer: {files, items}
    } as any;
}

function dataTransferItemFromFile(file: File): DataTransferItem {
    return {
        kind: 'file',
        type: file.type,
        getAsFile() {
            return file;
        },
        // tslint:disable-next-line: no-empty
        getAsString() {}
    } as any;
}

function dataTransferItem(file?: any, kind?: string, type: string = ''): DataTransferItem {
    return {
        kind,
        type,
        getAsFile() {
            return file;
        }
    } as any;
}

function dataTransferItemFromStr(str: string): DataTransferItem {
    return {
        kind: 'string',
        type: 'text/plain',
        getAsFile() {
            return null;
        },
        getAsString(cb: (data: string) => void) {
            return cb(str);
        }
    } as any;
}

function dataTransferItemFromEntry(entry: FileEntry | DirEntry, file?: File): DataTransferItem {
    return {
        kind: 'file',
        getAsFile() {
            return file;
        },
        webkitGetAsEntry: () => {
            return entry;
        }
    } as any;
}

function fileSystemFileEntryFromFile(file: File, err?: any): FileEntry {
    return {
        isDirectory: false,
        isFile: true,
        file(cb, errCb) {
            if (err) {
                errCb(err);
            } else {
                cb(file);
            }
        }
    };
}

function fileSystemDirEntryFromFile(
    files: FileOrDirEntry[],
    batchSize: number = 1,
    throwAfter: number = 0
): DirEntry {
    const copy = files.slice(0);
    const batches: FileOrDirEntry[][] = [];

    let current = 0;
    while (copy.length) {
        const length = copy.length;
        current += batchSize;
        const batch = copy.splice(0, current > length ? length : current);
        batches.push(batch);
    }

    return {
        isDirectory: true,
        isFile: false,
        createReader: () => {
            let cbCount = 0;

            return {
                readEntries(cb, errCb) {
                    const batch = batches[cbCount];
                    cbCount++;

                    if (throwAfter !== 0 && cbCount === throwAfter) {
                        errCb('Failed to read files');
                    }

                    if (batch) {
                        cb(batch);
                    } else {
                        cb([]);
                    }
                }
            };
        }
    };
}

function inputEvtFromFiles(...files: File[]): Event {
    const input = document.createElement('input');
    if (files.length) {
        Object.defineProperty(input, 'files', {
            value: files
        });
    }
    return {
        target: input
    } as any;
}

function createFile<T>(name: string, data: T, options?: FilePropertyBag) {
    const json = JSON.stringify(data);
    const file = new File([json], name, options);
    return file;
}

function sortFiles<T extends File>(files: T[]) {
    return files.slice(0)
        .sort((a, b) => a.name.localeCompare(b.name));
}


type FileOrDirEntry = FileEntry | DirEntry

interface FileEntry extends Entry {
    file(
        cb: (file: File) => void,
        errCb: (err: any) => void
    ): void;
}

interface DirEntry extends Entry {
    createReader(): DirReader;
}

interface Entry {
    isDirectory: boolean;
    isFile: boolean;
}

interface DirReader {
    readEntries(
        cb: (entries: FileOrDirEntry[]) => void,
        errCb: (err: any) => void
    ): void;
}
